from dataclasses import dataclass, field, fields

from fairseq.dataclass import FairseqDataclass
from fairseq.models.transformer import TransformerConfig
from fairseq.models.transformer.transformer_config import DecoderConfig, EncDecBaseConfig, QuantNoiseConfig
from fairseq.utils import safe_hasattr, safe_getattr

from .contextual_decoder_config import ContextualDecoderConfig


@dataclass
class CachingConfig(FairseqDataclass):
    propagate_context_encoder_gradient: bool = field(
        default=False,
        metadata={'help': 'propagate gradient to the encoder for context sentences'}
    )
    aggregate_sentence: bool = field(
        default=False,
        metadata={'help': 'aggregate tokens from the sentence into a single sentence representation'}
    )
    use_current_context: bool = field(
        default=False,
        metadata={'help': 'use current sentence as context (only applies when aggregate-sentence is True)'}
    )


@dataclass
class CachingContextualTransformerConfig(TransformerConfig):
    caching: CachingConfig = CachingConfig()
    context: ContextualDecoderConfig = ContextualDecoderConfig()

    @classmethod
    def from_namespace(cls, args):
        if args is None:
            return None
        if not isinstance(args, cls):
            seen = set()
            config = cls()
            # currently, we can go generically from DC fields to args hierarchically
            # but we can't easily deconstruct a flat namespace to a hierarchical
            # DC. Mostly because we could have a sub-dc called `decoder-foo` that should not
            # go to the sub struct called `decoder`. There are ways to go around this, but let's keep it simple
            # for now.
            for fld in fields(cls):
                # concretelly, the transformer_config know what sub-dc it has, so we go through all the dc fields
                # and if it's one that has a sub-dc, we build that sub-dc with `copy_keys()`
                if fld.name == "decoder":
                    if safe_hasattr(args, "decoder"):
                        #  in some cases, the args we receive is already structured (as DictConfigs), so let's just build the correct DC
                        seen.add("decoder")
                        config.decoder = DecoderConfig(**args.decoder)
                    else:
                        config.decoder = cls._copy_keys(
                            args, DecoderConfig, "decoder", seen
                        )
                elif fld.name == "encoder":
                    # same but for encoder
                    if safe_hasattr(args, "encoder"):
                        seen.add("encoder")
                        config.encoder = EncDecBaseConfig(**args.encoder)
                    else:
                        config.encoder = cls._copy_keys(
                            args, EncDecBaseConfig, "encoder", seen
                        )
                elif fld.name == "quant_noise":
                    # same but for quant_noise
                    if safe_hasattr(args, "quant_noise"):
                        seen.add("quant_noise")
                        config.quant_noise = QuantNoiseConfig(**args.quant_noise)
                    else:
                        config.quant_noise = cls._copy_keys(
                            args, QuantNoiseConfig, "quant_noise", seen
                        )
                elif fld.name == "caching":  # Added for shortening config
                    # same but for shortening
                    if safe_hasattr(args, "caching"):
                        seen.add("caching")
                        # config.shortening = ShorteningConfig(**args.shortening)
                    else:
                        config.caching = cls._copy_keys(
                            args, CachingConfig, "caching", seen
                        )
                elif fld.name == "context":  # Added for shortening config
                    # same but for shortening
                    if safe_hasattr(args, "context"):
                        seen.add("context")
                        # config.shortening = ShorteningConfig(**args.shortening)
                    else:
                        config.context = cls._copy_keys(
                            args, ContextualDecoderConfig, "context", seen
                        )
                elif safe_hasattr(args, fld.name):
                    # if it's not a structure field, it's just a normal field, copy it over
                    seen.add(fld.name)
                    setattr(config, fld.name, safe_getattr(args, fld.name))
            # we got all the fields defined in the dataclass, but
            # the argparse namespace might have extra args for two reasons:
            #   - we are in a legacy class so all the args are not declared in the dataclass. Ideally once everyone has defined a dataclass for their model, we won't need this
            #   - some places expect args to be there but never define them
            args_dict = (
                args._asdict()
                if safe_hasattr(args, "_asdict")
                else vars(args)
                if safe_hasattr(args, "__dict__")
                else {}
            )  # namedtupled doesn't have __dict__ :-/
            for key, value in args_dict.items():
                if key not in seen:
                    setattr(config, key, value)

            return config
        else:
            return args
