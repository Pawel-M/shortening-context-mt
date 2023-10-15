# From: https://github.com/neulab/contextual-mt
import warnings
from typing import Optional, List, Dict, Any

import torch
from fairseq.distributed import fsdp_wrap
from fairseq.models.transformer import (
    TransformerDecoder
)
from fairseq.modules import (
    TransformerDecoderLayer,
    LayerNorm,
    PositionalEmbedding
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from torch import Tensor


class TransformerDecoderLayerWrapper(TransformerDecoderLayer):
    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            context: Optional[torch.Tensor] = None,
            context_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
    ):
        return super(TransformerDecoderLayerWrapper, self).forward(
            x,
            encoder_out,
            encoder_padding_mask,
            incremental_state,
            prev_self_attn_state,
            prev_attn_state,
            self_attn_mask,
            self_attn_padding_mask,
            need_attn,
            need_head_weights,
        )


class ContextualTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(
            self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super(ContextualTransformerDecoderLayer, self).__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)
        self.parallel_cross_attention = args.context.use_parallel_cross_attention
        self.gated_attentions = args.context.use_gated_attentions

        self.context_gate = torch.nn.Sequential(
            torch.nn.Linear(self.embed_dim, 1),
            torch.nn.Sigmoid(),
        ) if args.context.use_gated_context_attention else None

        self.attentions_gate = torch.nn.Sequential(
            torch.nn.Linear(self.embed_dim * 2, 1),
            torch.nn.Sigmoid(),
        ) if args.context.use_gated_attentions else None


        self.context_attn = self.build_encoder_attention(self.embed_dim, args)
        self.context_attn_layer_norm = LayerNorm(self.embed_dim, export=args.export)

    def forward(
            self,
            x,
            encoder_out: Optional[torch.Tensor] = None,
            encoder_padding_mask: Optional[torch.Tensor] = None,
            context: Optional[torch.Tensor] = None,
            context_padding_mask: Optional[torch.Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            prev_self_attn_state: Optional[List[torch.Tensor]] = None,
            prev_attn_state: Optional[List[torch.Tensor]] = None,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            need_attn: bool = False,
            need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)

        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)

        if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        if self.c_attn is not None:
            tgt_len, bsz = x.size(0), x.size(1)
            x = x.view(tgt_len, bsz, self.nh, self.head_dim)
            x = torch.einsum("tbhd,h->tbhd", x, self.c_attn)
            x = x.reshape(tgt_len, bsz, self.embed_dim)
        if self.attn_ln is not None:
            x = self.attn_ln(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        self_attention_x = x

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        if self.context_attn is not None and context is not None:
            if self.parallel_cross_attention:
                cross_attention_x = x
                x = self_attention_x

            residual = x
            if self.normalize_before:
                x = self.context_attn_layer_norm(x)
            if prev_attn_state is not None:
                raise Exception('This part is not implemented!!!!!!!!!!!!!!!!!!!!!')
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, ctx_attn = self.context_attn(
                query=x,
                key=context,
                value=context,
                key_padding_mask=context_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)

            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.context_attn_layer_norm(x)

            if self.context_gate is not None:
                gate_value = self.context_gate(x)
                x = x * gate_value

            if self.parallel_cross_attention:
                if self.attentions_gate is not None:
                    gate_value = self.attentions_gate(torch.concat((x, cross_attention_x), dim=-1))
                    x = x * (1 - gate_value) + cross_attention_x * gate_value
                else:
                    x = x + cross_attention_x

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, ctx_attn, None


class ContextualTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, output_projection=None, ):
        self.args = args
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, output_projection)

        self.embed_dim = args.encoder.embed_dim
        self.concatenate_context = args.context.concatenate_context
        # self.coword_dropout = args.shortening.coword_dropout
        self.mask_id = dictionary.index("<mask>")

        self.sentence_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                self.embed_dim,
                padding_idx=0,
                learned=args.context.sentence_learned_pos,
            )
            if not args.context.no_sentence_positional_embeddings
            else None
        )
        self.sentence_repr_for_positions = torch.ones((1, args.max_source_positions))
        self.cached_sentence_pos = None

        # warnings.warn(f'sentence_positions: {self.sentence_positions.weights.shape}')

    def build_decoder_layer(self, args, no_encoder_attn=False):
        if args.context.concatenate_context:
            layer = TransformerDecoderLayerWrapper(args, no_encoder_attn)
        else:
            layer = ContextualTransformerDecoderLayer(args, no_encoder_attn)

        checkpoint = args.checkpoint_activations
        if checkpoint:
            offload_to_cpu = args.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = args.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
            self,
            prev_output_tokens,
            context_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            context_tokens (LongTensor): context tokens (ie a prefix
                to prev_output_tokens), shape `(batch, tgt_ctx_len)`
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            context_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
            self,
            prev_output_tokens,
            context_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            context_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. Aa copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            context_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Tensor = encoder_out["encoder_out"][0]
        padding_mask: Tensor = encoder_out["encoder_padding_mask"][0]

        context = encoder_out['context_out']
        context_padding_mask = encoder_out['context_padding_mask']

        if self.sentence_positions is not None:
            if self.training or self.cached_sentence_pos is None:
                self.sentence_repr_for_positions = self.sentence_repr_for_positions.to(enc.device)
                sentence_pos = self.sentence_positions(self.sentence_repr_for_positions)
                self.cached_sentence_pos = None
            else:
                sentence_pos = self.cached_sentence_pos

            context = [context[i] + sentence_pos[0, i] for i in range(len(context))]

        context_tensor = torch.cat(context, dim=1)
        context_padding_mask_tensor = torch.cat(context_padding_mask, dim=1)

        if self.concatenate_context:
            enc = torch.cat((context_tensor, enc), dim=1)
            padding_mask = torch.cat((context_padding_mask_tensor, padding_mask), dim=1)

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # Prevent torchscript exporting issue for dynamic quant embedding
        prev_output_tokens = prev_output_tokens.contiguous()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        enc = enc.transpose(0, 1)
        context_tensor = context_tensor.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # raise Exception(f'x: {x.shape}, enc: {enc.shape}, padding_mask: {padding_mask.shape}')
            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                context_tensor,
                context_padding_mask_tensor,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states, "encoder_out": encoder_out}

    def to(self, device):
        self.sentence_repr_for_positions = self.sentence_repr_for_positions.to(device)
        return super().to(device)

    # def drop_tokens(self, embeddings, padding_mask, coword_dropout_prob=0.0):
    #     if self.training and coword_dropout_prob > 0.0:
    #         mask_token = torch.tensor(self.mask_id).to(embeddings)
    #         probs = torch.ones_like(embeddings[:, :, 0]) * coword_dropout_prob
    #         mask = torch.logical_and(torch.bernoulli(probs), torch.logical_not(padding_mask))
    #         mask = torch.broadcast_to(torch.unsqueeze(mask, -1), embeddings.shape)
    #         embeddings = torch.where(mask == 0, embeddings, mask_token)
    #     return embeddings
