# From: https://github.com/neulab/contextual-mt

import torch


def encode(s, spm, vocab, return_tokenized=False):
    """binarizes a sentence according to sentencepiece model and a vocab"""
    tokenized = " ".join(spm.encode(s, out_type=str))
    encoded = vocab.encode_line(tokenized, append_eos=False, add_if_not_exist=False)
    if return_tokenized:
        return encoded, tokenized
    return encoded


def decode(ids, spm, vocab, return_tokenized=False):
    """decodes ids into a sentence"""
    tokenized = vocab.string(ids)
    decoded = spm.decode(tokenized.split())
    if return_tokenized:
        return decoded, tokenized
    return decoded


def create_context(sentences, context_size, break_id=None, eos_id=None):
    """based on list of context sentences tensors, creates a context tensor"""
    context = []
    # TODO: check if there is a bug here when context_size > len(sentences)
    for s in sentences[max(len(sentences) - context_size, 0):]:
        if context and break_id is not None:
            context.append(torch.tensor([break_id]))
        context.append(s)
    if eos_id is not None:
        context.append(torch.tensor([eos_id]))
    return torch.cat(context) if context else torch.tensor([]).long()


def create_context_list(sentences, context_size, break_id=None, eos_id=None):
    """based on list of context sentences tensors, creates a context tensor"""
    eos = torch.Tensor([eos_id]).long()
    context = list([torch.tensor([eos]).long() for _ in range(context_size)])
    # TODO: check if there is a bug here when context_size > len(sentences)
    effective_context_size = min(len(sentences), context_size)
    for i, s in enumerate(sentences[max(len(sentences) - context_size, 0):]):
        context[effective_context_size - i - 1] = torch.cat([s, eos])
    return context


def parse_documents(srcs, refs, docids):
    # parse lines into list of documents
    documents = []
    prev_docid = None
    for src_l, tgt_l, idx in zip(srcs, refs, docids):
        if prev_docid != idx:
            documents.append([])
        prev_docid = idx
        documents[-1].append((src_l, tgt_l))
    return documents
