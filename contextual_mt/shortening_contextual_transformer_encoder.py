# From: https://github.com/neulab/contextual-mt
import math
from typing import Optional, List, Dict

import torch
from fairseq.distributed import fsdp_wrap
from fairseq.models.transformer import (
    TransformerEncoder
)
from fairseq.modules import (
    MultiheadAttention,
    LayerNorm,
    PositionalEmbedding,
    transformer_layer
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from torch import Tensor

# From Funnel-Transformer
from contextual_mt.stop_grad import StopGradLayer


def pool_tensor(tensor, mode: str = "mean", stride: int = 2) -> torch.Tensor:
    """Apply 1D pooling to a tensor of size [B x T (x H)]."""
    if tensor is None:
        return None

    # Do the pool recursively if tensor is a list or tuple of tensors.
    if isinstance(tensor, (tuple, list)):
        return type(tensor)(pool_tensor(x, mode=mode, stride=stride) for x in tensor)

    ndim = tensor.ndim
    if ndim == 2:
        tensor = tensor[:, None, :, None]
    elif ndim == 3:
        tensor = tensor[:, None, :, :]
    # Stride is applied on the second-to-last dimension.
    stride = (stride, 1)

    if mode == "mean":
        tensor = torch.nn.functional.avg_pool2d(tensor, stride, stride=stride, ceil_mode=True)
    elif mode == "max":
        tensor = torch.nn.functional.max_pool2d(tensor, stride, stride=stride, ceil_mode=True)
    elif mode == "min":
        tensor = -torch.nn.functional.max_pool2d(-tensor, stride, stride=stride, ceil_mode=True)
    else:
        raise NotImplementedError("The supported modes are 'mean', 'max' and 'min'.")

    if ndim == 2:
        return tensor[:, 0, :, 0]
    elif ndim == 3:
        return tensor[:, 0]
    return tensor


def shorten_tokens(encoder_tokens, padding_mask, stride, mode="mean"):
    tokens_filled = encoder_tokens * (~padding_mask[..., None])

    length = encoder_tokens.shape[-2]
    remaining_size = math.ceil(length / stride) * stride - length
    remaining = torch.zeros((encoder_tokens.shape[0], remaining_size, encoder_tokens.shape[-1]),
                            dtype=encoder_tokens.dtype,
                            device=encoder_tokens.device)
    tokens_filled = torch.cat((tokens_filled, remaining), dim=-2)

    pooled_tokens = pool_tensor(tokens_filled, mode, stride)
    pooled_padding_mask = pool_tensor(padding_mask.to(torch.float), "mean", stride).floor().to(torch.bool)

    groups = None
    return pooled_tokens, pooled_padding_mask, groups


def pool_linear_tokens(encoder_tokens, padding_mask, stride, linear_pooler):
    tokens_filled = encoder_tokens * (~padding_mask[..., None])

    length = encoder_tokens.shape[-2]
    num_strides = math.ceil(length / stride)
    remaining_size = num_strides * stride - length
    remaining = torch.zeros((encoder_tokens.shape[0], remaining_size, encoder_tokens.shape[-1]),
                            dtype=encoder_tokens.dtype,
                            device=encoder_tokens.device)
    tokens_filled = torch.cat((tokens_filled, remaining), dim=-2)
    tokens_filled = torch.reshape(tokens_filled, (tokens_filled.shape[0], num_strides, -1))
    pooled_tokens = linear_pooler(tokens_filled)
    pooled_padding_mask = pool_tensor(padding_mask.to(torch.float), "mean", stride).floor().to(torch.bool)

    groups = None
    return pooled_tokens, pooled_padding_mask, groups


def group_tokens(encoder_tokens, tokens_to_group, padding_mask, shortening_cat, shortening_function=None):
    groups = shortening_cat(encoder_tokens)
    if shortening_function is not None:
        groups = (groups * torch.logical_not(padding_mask[..., None])) + (-1e8 * padding_mask[..., None])

    groups = groups * torch.logical_not(padding_mask[..., None])
    weights = groups
    weighted_tokens = weights[..., None] * tokens_to_group[..., None, :]
    pooled_tokens = torch.sum(weighted_tokens, dim=-3)

    shortened_padding_mask = torch.zeros(pooled_tokens.shape[:-1],
                                         dtype=torch.bool,
                                         device=encoder_tokens.device)

    return pooled_tokens, shortened_padding_mask, groups

class ShorteningTransformerEncoder(TransformerEncoder):
    def __init__(self, cfg, dictionary, embed_tokens, return_fc=False):
        super(ShorteningTransformerEncoder, self).__init__(cfg, dictionary, embed_tokens, return_fc)
        self.n_groups = cfg.shortening.groups
        self.embed_dim = cfg.encoder.embed_dim
        self.propagate_context_encoder_gradient = cfg.shortening.propagate_context_encoder_gradient
        self.propagate_context_size_gradient = cfg.shortening.propagate_context_size_gradient
        self.selecting_groups = cfg.shortening.selecting_groups
        self.use_current_context = not cfg.shortening.no_current_context
        self.use_pooling = cfg.shortening.use_pooling
        self.pooling_type = cfg.shortening.pooling_type

        self.ffn_hidden = cfg.shortening.ffn_hidden

        if not self.use_pooling:
            softmax_dim = -2 if self.selecting_groups else -1
            if self.ffn_hidden is not None:
                self.shortening_cat = torch.nn.Sequential(
                    torch.nn.Linear(self.embed_dim, self.ffn_hidden, bias=True),
                    torch.nn.GELU() if cfg.activation_fn == 'gelu' else torch.nn.ReLU(),
                    torch.nn.Linear(self.ffn_hidden, self.n_groups, bias=True),
                )
            else:
                self.shortening_cat = torch.nn.Sequential(
                    torch.nn.Linear(self.embed_dim, self.n_groups, bias=True),
                )
            if cfg.shortening.use_sparsemax:
                from .sparsemax import Sparsemax
                shortening_cat_function = Sparsemax(dim=softmax_dim)
            else:
                shortening_cat_function = torch.nn.Softmax(dim=softmax_dim)

            if self.selecting_groups:
                self.shortening_cat_function = shortening_cat_function
            else:
                self.shortening_cat.append(shortening_cat_function)
                self.shortening_cat_function = None

            num_positions = self.n_groups

            self.group_positions = (
                PositionalEmbedding(
                    num_positions,
                    self.embed_dim,
                    padding_idx=0,
                    learned=cfg.shortening.group_learned_pos,
                )
                if not cfg.shortening.no_group_positional_embeddings
                else None
            )
            self.group_repr_for_positions = torch.ones((1, num_positions))
        else:
            if self.pooling_type == 'linear':
                self.linear_pooler = torch.nn.Linear(self.embed_dim * self.n_groups, self.embed_dim)

            self.group_positions = (
                PositionalEmbedding(
                    cfg.max_source_positions,
                    self.embed_dim,
                    padding_idx=0,
                    learned=cfg.shortening.group_learned_pos,
                )
                if not cfg.shortening.no_group_positional_embeddings
                else None
            )
            self.group_repr_for_positions = None

        self.shortening_attn = self.build_self_attention(self.embed_dim, cfg)
        self.shortening_layer_norm = LayerNorm(self.embed_dim)
        self.stop_grad_layer = StopGradLayer()

    def build_self_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            q_noise=cfg.quant_noise.pq,
            qn_block_size=cfg.quant_noise.pq_block_size,
            xformers_att_config=cfg.encoder.xformers_att_config,
        )

    def build_encoder_layer(self, cfg):
        layer = transformer_layer.TransformerEncoderLayerBase(cfg, return_fc=self.return_fc)

        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def shorten(self, encoder_tokens, tokens_to_group, padding_mask):
        if self.use_pooling:
            if self.pooling_type == 'linear':
                pooled_tokens, shortened_padding_mask, groups = pool_linear_tokens(encoder_tokens, padding_mask,
                                                                                   self.n_groups,
                                                                                   self.linear_pooler)
            else:
                pooled_tokens, shortened_padding_mask, groups = shorten_tokens(encoder_tokens, padding_mask,
                                                                               self.n_groups,
                                                                               self.pooling_type)

        else:
            pooled_tokens, shortened_padding_mask, groups = group_tokens(encoder_tokens, tokens_to_group, padding_mask,
                                                                         self.shortening_cat,
                                                                         self.shortening_cat_function)

        pooled_tokens = pooled_tokens.transpose(0, 1)
        tokens_to_group = tokens_to_group.transpose(0, 1)
        resulting_tokens, _ = self.shortening_attn(
            query=pooled_tokens,
            key=tokens_to_group,
            value=tokens_to_group,
            key_padding_mask=padding_mask,
            need_weights=False,
            attn_mask=None,
        )
        resulting_tokens += pooled_tokens
        resulting_tokens = resulting_tokens.transpose(0, 1)
        resulting_tokens = self.shortening_layer_norm(resulting_tokens)

        return resulting_tokens, shortened_padding_mask, groups

    def forward(
            self,
            src_tokens,
            src_lengths,
            src_ctx_tokens,
            src_ctx_lengths,
            cached_src_context=None,
            cached_src_context_padding_mask=None,
            src_sample_probs=None,
            token_embeddings: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
    ):
        # Embed source tokens
        def embed(tokens):
            encoder_padding_mask = tokens.eq(self.padding_idx)
            has_pads = (
                    torch.tensor(tokens.device.type == "xla") or encoder_padding_mask.any()
            )
            # Torchscript doesn't handle bool Tensor correctly, so we need to work around.
            if torch.jit.is_scripting():
                has_pads = torch.tensor(1) if has_pads else torch.tensor(0)

            x, encoder_embedding = self.forward_embedding(tokens, token_embeddings)

            # account for padding while computing the representation
            x = x * (
                    1 - encoder_padding_mask.unsqueeze(-1).type_as(x) * has_pads.type_as(x)
            )

            # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
            # `forward` so we use a dictionary instead.
            # TorchScript does not support mixed values so the values are all lists.
            # The empty list is equivalent to None.
            src_lengths = (
                src_tokens.ne(self.padding_idx)
                .sum(dim=1, dtype=torch.int32)
                .reshape(-1, 1)
                .contiguous()
            )
            return x, encoder_padding_mask, has_pads, encoder_embedding

        # Encode source tokens
        def encode(tokens, padding_mask, has_pads, layers, layer_norm):
            # B x T x C -> T x B x C
            x = tokens.transpose(0, 1)

            encoder_states = []
            fc_results = []

            if return_all_hiddens:
                encoder_states.append(x)

            # encoder layers
            for layer in layers:
                lr = layer(
                    x=x,
                    encoder_padding_mask=padding_mask if has_pads else None,
                )

                if isinstance(lr, tuple) and len(lr) == 2:
                    x, fc_result = lr
                else:
                    x = lr
                    fc_result = None

                if return_all_hiddens and not torch.jit.is_scripting():
                    assert encoder_states is not None
                    encoder_states.append(x)
                    fc_results.append(fc_result)

            if layer_norm is not None:
                x = layer_norm(x)

            # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
            # `forward` so we use a dictionary instead.
            # TorchScript does not support mixed values so the values are all lists.
            # The empty list is equivalent to None.
            src_lengths = (
                src_tokens.ne(self.padding_idx)
                .sum(dim=1, dtype=torch.int32)
                .reshape(-1, 1)
                .contiguous()
            )

            x = x.transpose(0, 1)
            return x, encoder_states

        def encode_sentence(propagate_gradient_pre, propagate_gradient_post,
                            perform_shortening,
                            tokens):
            embed_x, embed_padding_mask, has_pads, embedding = embed(tokens)
            embed_x, embed_states = encode(
                tokens=embed_x,
                padding_mask=embed_padding_mask,
                has_pads=has_pads,
                layers=self.layers,
                layer_norm=self.layer_norm,
            )
            if not propagate_gradient_pre:
                embed_x = self.stop_grad_layer(embed_x)

            if perform_shortening:
                tokens_to_group = embed_x
                x, padding_mask, groups = self.shorten(
                    embed_x, tokens_to_group, embed_padding_mask
                )

                if self.group_positions is not None:
                    if self.group_repr_for_positions is not None:
                        self.group_repr_for_positions = self.group_repr_for_positions.to(x.device)
                        positions = self.group_repr_for_positions
                    else:
                        positions = torch.ones(x.shape[0:2], device=x.device)
                    group_pos = self.group_positions(positions) * (~padding_mask)[..., None]
                    x = x + group_pos

                if not propagate_gradient_post:
                    x = self.stop_grad_layer(x)

            else:
                x = embed_x
                padding_mask = embed_padding_mask
                groups = None

            return x, padding_mask, embed_states, embed_x, embed_padding_mask, embedding, groups

        context_out = []
        context_padding_mask = []
        # context_states = []
        all_groups = []
        ctx_x = None
        ctx_encoder_padding_mask = None

        if cached_src_context is not None and cached_src_context_padding_mask is not None:
            context_out = cached_src_context
            context_padding_mask = cached_src_context_padding_mask
        else:
            ctx_size = len(src_ctx_tokens)
            for ctx_idx, (src_ctx, ctx_len) in enumerate(zip(src_ctx_tokens, src_ctx_lengths)):
                propagate_context_gradient = (self.propagate_context_size_gradient is None
                                              or self.propagate_context_size_gradient >= ctx_size - ctx_idx)
                ctx_x, ctx_encoder_padding_mask, ctx_encoder_states, _, _, _, groups = encode_sentence(
                    propagate_gradient_pre=self.propagate_context_encoder_gradient,
                    propagate_gradient_post=propagate_context_gradient,
                    perform_shortening=True,
                    tokens=src_ctx, )

                context_out.append(ctx_x)
                context_padding_mask.append(ctx_encoder_padding_mask)
                # context_states.append(ctx_encoder_states)
                if groups is not None:
                    all_groups.append(groups)

        propagate_context_gradient = (self.propagate_context_size_gradient is None
                                      or self.propagate_context_size_gradient >= 0)

        shortened_x, shortened_padding_mask, encoder_states, \
        x, encoder_padding_mask, encoder_embedding, groups = encode_sentence(
            propagate_gradient_pre=True,
            propagate_gradient_post=propagate_context_gradient,
            perform_shortening=self.use_current_context,
            tokens=src_tokens
        )

        if self.use_current_context:
            context_out.append(shortened_x)
            context_padding_mask.append(shortened_padding_mask)
            # context_states.append(shortened_encoder_states)
            if groups is not None:
                all_groups.append(groups)

        return {
            "encoder_out": [x],  # B x T x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[List[T x B x C]]
            "context_out": context_out,  # List[B x G x C]
            "context_padding_mask": context_padding_mask,  # List[B x G]
            # "context_states": context_states,  # List[List[G x B x C]]
            "src_tokens": [src_tokens],
            "src_lengths": [src_lengths],
            "src_ctx_tokens": src_ctx_tokens,
            "src_ctx_lengths": src_ctx_lengths,
            "groups": all_groups,
        }

    def get_out_string(self, encoder_out):
        s = '{\n'
        for k in encoder_out:
            v = encoder_out[k]
            if isinstance(v, list):
                if len(v) > 0:
                    v = v[0].shape
            else:
                v = v.shape

            s += f'{k}: {v}\n'
        s += '\n'
        return s

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """

        # Encoder outputs
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(0, new_order)]

        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [encoder_out["encoder_padding_mask"][0].index_select(0, new_order)]

        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [encoder_out["encoder_embedding"][0].index_select(0, new_order)]

        new_encoder_states = encoder_out["encoder_states"]
        if len(new_encoder_states) > 0:
            for idx, state in enumerate(new_encoder_states):
                new_encoder_states[idx] = state.index_select(1, new_order)

        # Context outputs
        if len(encoder_out["context_out"]) == 0:
            new_context_out = []
        else:
            new_context_out = [v.index_select(0, new_order) for v in encoder_out["context_out"]]

        if len(encoder_out["context_padding_mask"]) == 0:
            new_context_padding_mask = []
        else:
            new_context_padding_mask = [v.index_select(0, new_order) for v in encoder_out["context_padding_mask"]]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        if len(encoder_out["src_ctx_tokens"]) == 0:
            src_ctx_tokens = []
        else:
            src_ctx_tokens = [
                c.index_select(0, new_order)
                for c in encoder_out['src_ctx_tokens']
            ]

        if len(encoder_out["src_ctx_lengths"]) == 0:
            src_ctx_lengths = []
        else:
            src_ctx_lengths = [
                c.index_select(0, new_order)
                for c in encoder_out["src_ctx_lengths"]
            ]

        if len(encoder_out['groups']) == 0:
            new_groups = []
        else:
            new_groups = [v.index_select(0, new_order) for v in encoder_out['groups']]

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": new_encoder_states,  # List[T x B x C]
            "context_out": new_context_out,
            "context_padding_mask": new_context_padding_mask,
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
            "src_ctx_tokens": src_ctx_tokens,
            "src_ctx_lengths": src_ctx_lengths,
            "groups": new_groups,
        }
