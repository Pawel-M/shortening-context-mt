from typing import Optional, List, Dict

import torch
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerEncoder
)
from fairseq.models.transformer import (
    TransformerModel,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP
)
from fairseq.models.transformer import (
    base_architecture as transformer_base_architecture,
    transformer_iwslt_de_en,
    transformer_vaswani_wmt_en_de_big,
)
from torch import Tensor

from .caching_contextual_transformer_config import CachingContextualTransformerConfig
from .contextual_decoder import ContextualTransformerDecoder
from .stop_grad import StopGradLayer


class CachingTransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens, return_fc=False):
        self.args = args
        super().__init__(
            CachingContextualTransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            return_fc=return_fc,
        )

        cfg = self.cfg

        self.embed_dim = cfg.encoder.embed_dim
        self.propagate_context_encoder_gradient = cfg.caching.propagate_context_encoder_gradient
        self.use_current_context = cfg.caching.use_current_context
        self.aggregate_sentence = cfg.caching.aggregate_sentence

        self.stop_grad_layer = StopGradLayer()

    def build_encoder_layer(self, args):
        return super().build_encoder_layer(
            CachingContextualTransformerConfig.from_namespace(args),
        )

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
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(
            src_tokens,
            src_lengths,
            src_ctx_tokens,
            src_ctx_lengths,
            cached_src_context,
            cached_src_context_padding_mask,
            src_sample_probs,
            token_embeddings,
            return_all_hiddens,
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
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
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """

        def embed(tokens):
            # compute padding mask
            padding_mask = tokens.eq(self.padding_idx)
            has_pads = tokens.device.type == "xla" or padding_mask.any()

            x, embedding = self.forward_embedding(tokens, token_embeddings)

            # account for padding while computing the representation
            if has_pads:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

            return x, embedding, padding_mask, has_pads

        def encode_sentence(tokens, states, fc_results):
            x, embedding, padding_mask, has_pads = embed(tokens)

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

            if return_all_hiddens:
                states.append(x)

            # encoder layers
            for layer in self.layers:
                lr = layer(
                    x, encoder_padding_mask=padding_mask if has_pads else None
                )

                if isinstance(lr, tuple) and len(lr) == 2:
                    x, fc_result = lr
                else:
                    x = lr
                    fc_result = None

                if return_all_hiddens and not torch.jit.is_scripting():
                    assert states is not None
                    states.append(x)
                    fc_results.append(fc_result)

            if self.layer_norm is not None:
                x = self.layer_norm(x)

            # T x B x C -> B x T x C
            x = x.transpose(0, 1)
            return x, embedding, padding_mask

        context_out = []
        context_padding_mask = []
        context_states = []
        context_fc_results = []

        if cached_src_context is not None and cached_src_context_padding_mask is not None:
            context_out = cached_src_context
            context_padding_mask = cached_src_context_padding_mask
        else:
            for src_ctx in src_ctx_tokens:
                ctx_states = []
                ctx_fc_results = []
                ctx_x, ctx_embedding, ctx_padding_mask = encode_sentence(
                    src_ctx,
                    ctx_states,
                    ctx_fc_results,
                )

                if self.aggregate_sentence:
                    ctx_x = ctx_x * torch.logical_not(torch.unsqueeze(ctx_padding_mask, dim=-1))
                    ctx_x = torch.unsqueeze(torch.mean(ctx_x, dim=1), dim=1)
                    ctx_padding_mask = ctx_padding_mask[:, 0:1]

                if not self.propagate_context_encoder_gradient:
                    ctx_x = self.stop_grad_layer(ctx_x)

                context_out.append(ctx_x)
                context_padding_mask.append(ctx_padding_mask)
                context_states.append(ctx_states)
                context_fc_results.append(ctx_fc_results)

        encoder_states = []
        encoder_fc_results = []
        x, encoder_embedding, encoder_padding_mask = encode_sentence(src_tokens, encoder_states, encoder_fc_results)

        if self.use_current_context:
            # include current sentence
            if self.aggregate_sentence:
                ctx_x = x * torch.logical_not(torch.unsqueeze(encoder_padding_mask, dim=-1))
                ctx_x = torch.unsqueeze(torch.mean(ctx_x, dim=1), dim=1)
                ctx_padding_mask = encoder_padding_mask[:, 0:1]
                context_out.append(ctx_x)
                context_padding_mask.append(ctx_padding_mask)
            else:
                context_out.append(x)
                context_padding_mask.append(encoder_padding_mask)

            context_states.append(encoder_states)
            context_fc_results.append(encoder_fc_results)

        return {
            "encoder_out": [x],  # B x T x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[List[T x B x C]]
            # "encoder_fc_results": encoder_fc_results,
            "context_out": context_out,  # List[B x G x C]
            "context_padding_mask": context_padding_mask,  # List[B x G]
            "context_states": context_states,  # List[List[G x B x C]]
            # "context_fc_results": context_fc_results,
            "src_tokens": [src_tokens],
            "src_lengths": [src_lengths],
            "src_ctx_tokens": src_ctx_tokens,
            "src_ctx_lengths": src_ctx_lengths,
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

        new_context_states = encoder_out["context_states"]
        if len(new_context_states) > 0:
            for idx, state in enumerate(new_context_states):
                new_context_states[idx] = [s.index_select(1, new_order) for s in state]

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

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": new_encoder_states,  # List[T x B x C]
            "context_out": new_context_out,
            "context_padding_mask": new_context_padding_mask,
            "context_states": new_context_states,
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
            "src_ctx_tokens": src_ctx_tokens,
            "src_ctx_lengths": src_ctx_lengths,
            # "groups": new_groups,
        }


@register_model("caching_contextual_transformer")
class CachingContextualTransformerModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        gen_parser_from_dataclass(
            parser,
            CachingContextualTransformerConfig(),
            delete_default=True,
            with_prefix=""
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        # base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            args.share_decoder_input_output_embed = True

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        if not args.share_all_embeddings:
            args.min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
        cfg = CachingContextualTransformerConfig.from_namespace(args)
        return super().build_model(cfg, task)

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return CachingTransformerEncoder(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return ContextualTransformerDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.cfg = CachingContextualTransformerConfig.from_namespace(args)
        print(f'config: {self.cfg}')

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            src_ctx_tokens=None,
            src_ctx_lengths=None,
            tgt_ctx_tokens=None,
            tgt_ctx_lengths=None,
            cached_src_context=None,
            cached_src_context_padding_mask=None,
            src_sample_probs=None,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.
        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            src_ctx_tokens=src_ctx_tokens,
            src_ctx_lengths=src_ctx_lengths,
            cached_src_context=cached_src_context,
            cached_src_context_padding_mask=cached_src_context_padding_mask,
            src_sample_probs=src_sample_probs,
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            context_tokens=tgt_ctx_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out


@register_model_architecture("caching_contextual_transformer", "caching_contextual_transformer")
def caching_contextual_transformer_base_architecture(args):
    transformer_base_architecture(args)


@register_model_architecture("caching_contextual_transformer", "caching_contextual_transformer_iwslt")
def caching_contextual_transformer_iwslt_architecture(args):
    transformer_iwslt_de_en(args)


@register_model_architecture("caching_contextual_transformer", "caching_contextual_transformer_big")
def caching_contextual_transformer_big_architecture(args):
    transformer_vaswani_wmt_en_de_big(args)
