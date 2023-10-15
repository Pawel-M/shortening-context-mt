# Adapted fromrom: https://github.com/neulab/contextual-mt

from typing import Optional

from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import (
    register_model,
    register_model_architecture,
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

from .contextual_decoder import ContextualTransformerDecoder
from .shortening_contextual_transformer_config import ShorteningContextualTransformerConfig
from .shortening_contextual_transformer_encoder import ShorteningTransformerEncoder


@register_model("shortening_contextual_transformer")
class ShorteningContextualTransformerModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        gen_parser_from_dataclass(
            parser,
            ShorteningContextualTransformerConfig(),
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

        # print('===============')
        # print(args)
        # print('===============')
        if 'shortening_sentence_learned_pos' in args:
            args.context_sentence_learned_pos = args.shortening_sentence_learned_pos
        if 'shortening_no_sentence_positional_embeddings' in args:
            args.context_no_sentence_positional_embeddings = args.shortening_no_sentence_positional_embeddings
        if 'shortening_concatenate_context' in args:
            args.context_concatenate_context = args.shortening_concatenate_context

        cfg = ShorteningContextualTransformerConfig.from_namespace(args)

        model = super().build_model(cfg, task)
        return model

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return ShorteningTransformerEncoder(cfg, src_dict, embed_tokens)

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
        self.cfg = ShorteningContextualTransformerConfig.from_namespace(args)
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


@register_model_architecture("shortening_contextual_transformer", "shortening_contextual_transformer")
def shortening_contextual_transformer_base_architecture(args):
    transformer_base_architecture(args)


@register_model_architecture("shortening_contextual_transformer", "shortening_contextual_transformer_iwslt")
def shortening_contextual_transformer_iwslt_architecture(args):
    transformer_iwslt_de_en(args)


@register_model_architecture("shortening_contextual_transformer", "shortening_contextual_transformer_big")
def shortening_contextual_transformer_big_architecture(args):
    transformer_vaswani_wmt_en_de_big(args)
