from dataclasses import dataclass, field

from fairseq.dataclass import FairseqDataclass

@dataclass
class ContextualDecoderConfig(FairseqDataclass):
    no_sentence_positional_embeddings: bool = field(
        default=False,
        metadata={'help': 'do not use sentence positional embeddings'}
    )
    sentence_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned sentence positional embeddings"}
    )
    concatenate_context: bool = field(
        default=False,
        metadata={'help': 'concatenate context with the encoder outputs'}
    )
    use_parallel_cross_attention: bool = field(
        default=False,
        metadata={'help': 'use parallel cross-attention '
                          '(only if using separate cross-attention for context and not concatenating context)'}
    )
    use_gated_context_attention: bool = field(
        default=False,
        metadata={'help': 'use gate for the context attention'}
    )
    use_gated_attentions: bool = field(
        default=False,
        metadata={'help': 'use gate for the context and encoder attentions (only when parallel attentions are used)'}
    )
