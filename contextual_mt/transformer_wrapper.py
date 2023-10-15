from typing import Optional, Dict, List, Any

import torch
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerModel, TransformerConfig, TransformerEncoderBase, TransformerDecoderBase,
)

import fairseq.models.transformer
from torch import Tensor


class TransformerEncoderWrapper(TransformerEncoderBase):
    def forward(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
            src_ctx_tokens=None,
            src_ctx_lengths=None,
            src_sample_probs=None,
    ):
        return super(TransformerEncoderWrapper, self).forward(
            src_tokens,
            src_lengths,
            return_all_hiddens,
            token_embeddings)


class TransformerDecoderWrapper(TransformerDecoderBase):
    def forward(self,
                prev_output_tokens,
                context_tokens=None,
                encoder_out: Optional[Dict[str, List[Tensor]]] = None,
                incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
                features_only: bool = False,
                full_context_alignment: bool = False,
                alignment_layer: Optional[int] = None,
                alignment_heads: Optional[int] = None,
                src_lengths: Optional[Any] = None,
                return_all_hiddens: bool = False, ):
        return super(TransformerDecoderWrapper, self).forward(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            features_only,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            src_lengths,
            return_all_hiddens
        )


@register_model("transformer_context_wrapper")
class TransformerWrapperModel(TransformerModel):
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoderWrapper(
            TransformerConfig.from_namespace(args), src_dict, embed_tokens
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoderWrapper(
            TransformerConfig.from_namespace(args), tgt_dict, embed_tokens
        )

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
            src_sample_probs=None,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        return super(TransformerWrapperModel, self).forward(src_tokens,
                                                            src_lengths,
                                                            prev_output_tokens,
                                                            return_all_hiddens,
                                                            features_only,
                                                            alignment_layer,
                                                            alignment_heads)


@register_model_architecture("transformer_context_wrapper", "transformer_wrapper_tiny")
def transformer_wrapper_tiny_architecture(args):
    fairseq.models.transformer.tiny_architecture(args)


@register_model_architecture("transformer_context_wrapper", "transformer_wrapper")
def transformer_wrapper_base_architecture(args):
    fairseq.models.transformer.base_architecture(args)


@register_model_architecture("transformer_context_wrapper", "transformer_wrapper_iwslt_de_en")
def transformer_wrapper_iwslt_de_en(args):
    fairseq.models.transformer.transformer_iwslt_de_en(args)


@register_model_architecture("transformer_context_wrapper", "transformer_wrapper_wmt_en_de")
def transformer_wrapper_wmt_en_de(args):
    fairseq.models.transformer.transformer_wmt_en_de(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("transformer_context_wrapper", "transformer_wrapper_vaswani_wmt_en_de_big")
def transformer_wrapper_vaswani_wmt_en_de_big(args):
    fairseq.models.transformer.transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture("transformer_context_wrapper", "transformer_wrapper_vaswani_wmt_en_fr_big")
def transformer_wrapper_vaswani_wmt_en_fr_big(args):
    fairseq.models.transformer.transformer_vaswani_wmt_en_fr_big(args)


@register_model_architecture("transformer_context_wrapper", "transformer_wrapper_wmt_en_de_big")
def transformer_wrapper_wmt_en_de_big(args):
    fairseq.models.transformer.transformer_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture("transformer_context_wrapper", "transformer_wrapper_wmt_en_de_big_t2t")
def transformer_wrapper_wmt_en_de_big_t2t(args):
    fairseq.models.transformer.transformer_wmt_en_de_big_t2t(args)
