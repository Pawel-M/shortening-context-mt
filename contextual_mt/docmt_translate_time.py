# From: https://github.com/neulab/contextual-mt

import argparse
import os
import pickle
import time
import warnings
from copy import deepcopy

import numpy as np
import tqdm

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from fairseq import utils, hub_utils

import sentencepiece as sp

import contextual_mt  # noqa: F401
from contextual_mt.contextual_dataset import collate as contextual_collate
from contextual_mt.contextual_list_dataset import collate as contextual_list_collate
from contextual_mt import ContextualSequenceGenerator

from contextual_mt.utils import encode, decode, create_context, create_context_list, parse_documents


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", required=True, help="file to be translated")
    parser.add_argument("--docids-file", required=True, help="file with document ids")
    parser.add_argument(
        "--predictions-file", required=True, help="file to save the predictions"
    )
    parser.add_argument("--results-file", default='results.csv', required=False, help="file to save the results")
    parser.add_argument(
        "--reference-file",
        default=None,
        help="reference file, used if with --gold-target-context",
    )
    parser.add_argument("--source-lang", default=None)
    parser.add_argument("--target-lang", default=None)
    parser.add_argument(
        "--path", required=True, metavar="FILE", help="path to model file"
    )
    parser.add_argument("--beam", default=5, type=int, metavar="N", help="beam size")
    parser.add_argument(
        "--max-len-a",
        default=0,
        type=float,
        metavar="N",
        help=(
            "generate sequences of maximum length ax + b, "
            "where x is the source length"
        ),
    )
    parser.add_argument(
        "--max-len-b",
        default=200,
        type=int,
        metavar="N",
        help=(
            "generate sequences of maximum length ax + b, "
            "where x is the source length"
        ),
    )
    parser.add_argument(
        "--min-len",
        default=1,
        type=float,
        metavar="N",
        help=("minimum generation length"),
    )
    parser.add_argument(
        "--lenpen",
        default=1,
        type=float,
        help="length penalty: <1.0 favors shorter, >1.0 favors longer sentences",
    )
    parser.add_argument(
        "--gold-target-context",
        default=False,
        action="store_true",
        help="if set, model will use ground-truth targets as context",
    )
    parser.add_argument(
        "--last-checkpoint",
        default=False,
        action="store_true",
        help="use the model's last checkpoint instead of the best",
    )
    parser.add_argument("--source-context-size", default=None, type=int)
    parser.add_argument("--target-context-size", default=None, type=int)
    parser.add_argument("--no-progress", default=False, action="store_true", help="do not show progress bar", )
    parser.add_argument("--separate-context-sentences", default=False, action="store_true",
                        help="if set, separates context sentences into a list", )
    parser.add_argument("--limit-size", default=None, type=int)
    parser.add_argument("--split-name", default=None, help='base name prepended to groups and tokens files')
    parser.add_argument("--use-cache", default=False, action="store_true")
    parser.add_argument("--last-context-as-cache", default=False, action="store_true",
                        help="use last context from the encoder as the representation to cache")
    args = parser.parse_args()

    return args


def load_model(path, args, best=True, use_cuda=True):
    checkpoint_file = 'checkpoint_best.pt' if best else 'checkpoint_last.pt'
    pretrained = hub_utils.from_pretrained(
        path, checkpoint_file=checkpoint_file
    )
    models = pretrained["models"]
    for model in models:
        if use_cuda:
            model.cuda()
        model.eval()

    # load dict, params and generator from task
    src_dict = pretrained["task"].src_dict
    tgt_dict = pretrained["task"].tgt_dict
    source_context_size = (
        pretrained["task"].args.source_context_size
        if args.source_context_size is None
        else args.source_context_size
    )
    target_context_size = (
        pretrained["task"].args.target_context_size
        if args.target_context_size is None
        else args.target_context_size
    )
    separate_context_sentences = (
        pretrained["task"].args.separate_context_sentences
        if args.separate_context_sentences is None
        else args.separate_context_sentences
    )

    return pretrained, models, src_dict, tgt_dict, source_context_size, target_context_size, separate_context_sentences


def load_spm(args):
    # load sentencepiece models (assume they are in the checkpoint dirs)
    if os.path.exists(os.path.join(args.path, "spm.model")):
        spm = sp.SentencePieceProcessor()
        spm.Load(os.path.join(args.path, "spm.model"))
        src_spm = spm
        tgt_spm = spm
    else:
        assert args.source_lang is not None and args.target_lang is not None
        src_spm = sp.SentencePieceProcessor()
        src_spm.Load(os.path.join(args.path, f"spm.{args.source_lang}.model"))
        tgt_spm = sp.SentencePieceProcessor()
        tgt_spm.Load(os.path.join(args.path, f"spm.{args.target_lang}.model"))

    return src_spm, tgt_spm


def load_documents(args):
    # load files needed
    with open(args.source_file, "r", encoding="utf-8") as src_f:
        srcs = [line.strip() for line in src_f]
    with open(args.docids_file, "r", encoding="utf-8") as docids_f:
        docids = [int(idx) for idx in docids_f]
    if args.reference_file is not None:
        with open(args.reference_file, "r", encoding="utf-8") as tgt_f:
            refs = [line.strip() for line in tgt_f]
    else:
        refs = [None for _ in srcs]

    documents = parse_documents(srcs, refs, docids)
    return srcs, docids, refs, documents


if __name__ == "__main__":
    args = load_args()
    print(args)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        peak_memory_no_model = torch.cuda.max_memory_allocated()

    (
        pretrained,
        models,
        src_dict,
        tgt_dict,
        source_context_size,
        target_context_size,
        separate_context_sentences
    ) = load_model(args.path, args, not args.last_checkpoint, torch.cuda.is_available())
    checkpoint_prefix = 'last' if args.last_checkpoint else 'best'

    use_cache = args.use_cache
    last_context_as_cache = args.last_context_as_cache

    if args.gold_target_context:
        assert args.reference_file is not None

    models = [model.eval() for model in models]

    task = pretrained["task"]
    generator = task.build_generator(
        models, args, seq_gen_cls=ContextualSequenceGenerator
    )

    model_name = pretrained['models'][0]._get_name()
    print('model_name', model_name)
    print('source_context_size', source_context_size)
    print('target_context_size', target_context_size)
    print('separate_context_sentences', separate_context_sentences)
    print('use_cache', use_cache)
    print('last_context_as_cache', last_context_as_cache)
    print('GPU:', torch.cuda.is_available())
    print('no progress bar:', args.no_progress)

    print('model parameters:')
    for n, p in models[0].named_parameters():
        print(n)

    src_spm, tgt_spm = load_spm(args)
    srcs, docids, refs, documents = load_documents(args)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        peak_memory_model_only = torch.cuda.max_memory_allocated()

    if args.limit_size is not None:
        size = min(args.limit_size, len(documents))
        documents = documents[:size]

    bar = tqdm.tqdm(total=int(np.sum([len(d) for d in documents]))) if not args.no_progress else None

    doc_lens = [len(document) for document in documents]
    preds = []
    document_prediction_times = []
    sentence_prediction_times = []
    peak_sentence_memory = []
    peak_encoding_memory = []
    num_input_tokens = []
    num_source_tokens = []
    num_source_and_context_tokens = []
    num_target_tokens = []
    encoder_times = []
    decoder_times = []
    for document in documents:
        src_context_lines = []
        cached_src_context = []
        cached_src_context_padding_masks = []

        document_translation_start_time = time.time()
        for src_sentence, tgt_sentence in document:
            source_noeos, tokens = encode(src_sentence, src_spm, src_dict, return_tokenized=True)
            # source_noeos = torch.cat([source_noeos], dim=0)
            source = torch.stack([*source_noeos, torch.tensor(src_dict.eos())])
            num_input_tokens.append(source.shape[0])
            num_source_tokens.append(source.shape[0])
            num_source_and_context_tokens.append(
                source.shape[0] + np.sum([ctx.shape[0] + 1 for ctx in src_context_lines]))
            sample = [{
                "id": 0,
                "source": source,
            }]

            if use_cache and len(cached_src_context) >= source_context_size:
                src_context = create_context_list([], 0, break_id=src_dict.index("<brk>"), eos_id=src_dict.eos(), )
                tgt_context = create_context_list([], 0, break_id=src_dict.index("<brk>"), eos_id=src_dict.eos(), )
                sample[0]['cached_src_context'] = cached_src_context
                sample[0]['cached_src_context_padding_mask'] = cached_src_context_padding_masks
                num_input_tokens[-1] = num_input_tokens[-1] + np.sum([c.shape[0] for c in cached_src_context])
            else:
                if separate_context_sentences:
                    src_context = create_context_list(
                        src_context_lines,
                        source_context_size,
                        break_id=src_dict.index("<brk>"),
                        eos_id=src_dict.eos(),
                    )
                    tgt_context = create_context_list([], 0, break_id=src_dict.index("<brk>"), eos_id=src_dict.eos(), )
                    num_input_tokens[-1] = num_input_tokens[-1] + np.sum([c.shape[0] for c in src_context])
                else:
                    src_context = create_context(
                        src_context_lines,
                        source_context_size,
                        break_id=src_dict.index("<brk>"),
                        eos_id=src_dict.eos(),
                    )
                    tgt_context = create_context([], 0, break_id=src_dict.index("<brk>"), eos_id=src_dict.eos(), )
                    num_input_tokens[-1] = num_input_tokens[-1] + src_context.shape[0]
            sample[0]['src_context'] = src_context
            sample[0]['tgt_context'] = tgt_context

            # create batch
            if separate_context_sentences:
                batch_sample = contextual_list_collate(sample, src_dict.pad(), src_dict.eos())
            else:
                batch_sample = contextual_collate(sample, src_dict.pad(), src_dict.eos())

            if torch.cuda.is_available():
                batch_sample = utils.move_to_cuda(batch_sample)

            sentence_translation_start_time = time.time()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                output = task.inference_step(generator, models, batch_sample)

            if torch.cuda.is_available():
                peak_sentence_memory.append(torch.cuda.max_memory_allocated())


            sentence_translation_finish_time = time.time()
            sentence_prediction_times.append(sentence_translation_finish_time - sentence_translation_start_time)
            encoder_times.append(generator.encoder_time)
            decoder_times.append(generator.decoder_time)

            # Encoding-only time
            net_input = deepcopy(batch_sample['net_input'])
            del net_input['tgt_ctx_tokens']
            del net_input['tgt_ctx_lengths']

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                enc_out = models[0].encoder.forward_torchscript(net_input)

            if torch.cuda.is_available():
                peak_encoding_memory.append(torch.cuda.max_memory_allocated())

            generator.encoder_time = 0
            generator.decoder_time = 0

            hyp_ids = output[0][0]["tokens"].cpu()
            num_target_tokens.append(hyp_ids.shape[0])
            decoded, decoded_tokens = decode(hyp_ids, tgt_spm, tgt_dict, return_tokenized=True)
            preds.append(decoded)

            src_context_lines.append(source_noeos)
            src_context_lines = src_context_lines[:source_context_size]

            if use_cache:
                encoder_out = output[0][0]['encoder_out']
                if last_context_as_cache:
                    cached_src_context.append(encoder_out['context_out'][-1][0].detach().cpu())
                    cached_src_context_padding_masks.append(encoder_out['context_padding_mask'][-1][0].detach().cpu())
                else:
                    cached_src_context.append(encoder_out['encoder_out'][0][0].detach().cpu())
                    cached_src_context_padding_masks.append(encoder_out['encoder_padding_mask'][0][0].detach().cpu())

                cached_src_context = cached_src_context[:source_context_size]
                cached_src_context_padding_masks = cached_src_context_padding_masks[:source_context_size]

            if bar is not None:
                bar.update(1)

        document_translation_finish_time = time.time()
        document_prediction_times.append(document_translation_finish_time - document_translation_start_time)

    if bar is not None:
        bar.close()

    # print(preds)
    # print(document_prediction_times)
    # print(sentence_prediction_times)
    mean_sentence_inference_time = np.mean(sentence_prediction_times)
    mean_document_inference_time = np.mean(document_prediction_times)
    mean_document_sentence_inference_time = np.mean([document_prediction_times[i] / doc_lens[i]
                                                     for i in range(len(document_prediction_times))])
    print('mean_sentence_inference_time', mean_sentence_inference_time)
    print('mean_document_inference_time', mean_document_inference_time)
    print('mean_document_sentence_inference_time', mean_document_sentence_inference_time)
    print('encoder time', generator.encoder_time)
    print('decoder time', generator.decoder_time)

    print(document_prediction_times)

    num_input_tokens = np.array(num_input_tokens)
    num_source_tokens = np.array(num_source_tokens)
    num_source_and_context_tokens = np.array(num_source_and_context_tokens)
    num_target_tokens = np.array(num_target_tokens)
    num_all_tokens = num_input_tokens + num_target_tokens
    mean_num_tokens = np.mean(num_input_tokens)
    total_source_tokens = np.sum(num_source_tokens)
    total_target_tokens = np.sum(num_target_tokens)
    total_all_tokens = total_source_tokens + total_target_tokens
    total_encoder_time = np.sum(encoder_times)
    total_decoder_time = np.sum(decoder_times)

    import pandas as pd

    df = pd.DataFrame(list(
        zip(num_source_tokens, num_source_and_context_tokens, num_input_tokens, num_target_tokens, num_all_tokens,
            sentence_prediction_times, encoder_times, decoder_times)),
        columns=['Source Tokens', 'Source and Context Tokens', 'Input Tokens', 'Target Tokens', 'Total Tokens',
                 'Prediction Times', 'Encoder Times', 'Decoder Times'])

    if torch.cuda.is_available():
        peak_sentence_memory = np.array(peak_sentence_memory)
        peak_sentence_memory_gb = peak_sentence_memory / (1024 ** 3)
        max_memory_allocated = np.max(peak_sentence_memory)
        max_memory_allocated_gb = max_memory_allocated / (1024 ** 3)
        mean_memory_allocated = np.mean(peak_sentence_memory)
        mean_memory_allocated_gb = mean_memory_allocated / (1024 ** 3)
        max_memory_per_token_gb = max_memory_allocated_gb / total_all_tokens
        mean_memory_per_token_gb = mean_memory_allocated_gb / total_all_tokens
        peak_memory_no_model_gb = peak_memory_no_model / (1024 ** 3)
        peak_memory_model_only_gb = peak_memory_model_only / (1024 ** 3)

        peak_encoding_memory = np.array(peak_encoding_memory)
        peak_encoding_memory_gb = peak_encoding_memory / (1024 ** 3)

        print('peak_sentence_memory [GB]', peak_sentence_memory_gb)
        print('peak_encoding_memory [GB]', peak_encoding_memory_gb)
        print('cuda max memory allocated [B]', max_memory_allocated)
        print('cuda max memory allocated [GB]', max_memory_allocated_gb)
        print('cuda mean memory allocated [GB]', mean_memory_allocated_gb)
        print('cuda no model [GB]', peak_memory_no_model_gb)
        print('cuda model only [GB]', peak_memory_model_only_gb)

        df['Memory [GB]'] = pd.DataFrame(peak_sentence_memory_gb)
        results_file_parts = args.results_file.split('.')
        results_file_parts[-2] = results_file_parts[-2] + '_gpu'
        results_file = '.'.join(results_file_parts)
    else:
        max_memory_allocated = ''
        max_memory_allocated_gb = ''
        mean_memory_allocated = ''
        mean_memory_allocated_gb = ''
        max_memory_per_token_gb = ''
        mean_memory_per_token_gb = ''
        peak_memory_no_model_gb = ''
        peak_memory_model_only_gb = ''

        results_file = args.results_file

    print('results file', results_file)
    df['Source Context Size'] = source_context_size
    df.to_csv(results_file, sep=',')

    print('mean num tokens', mean_num_tokens)
    print('mean num target tokens', np.mean(num_target_tokens))
    print('num sentences', np.sum(doc_lens))
    print('num source tokens', np.sum(num_source_tokens))
    print('num target tokens', np.sum(num_target_tokens))

    run = os.path.abspath(args.path).split('/')[-2]
    print(
        f'{run}\t{model_name}\t{source_context_size}\t{args.beam}'
        f'\t{mean_sentence_inference_time}\t{mean_document_inference_time}\t{mean_document_sentence_inference_time}'
        f'\t{total_encoder_time}\t{total_decoder_time}'
        f'\t{total_encoder_time / total_source_tokens}\t{total_decoder_time / total_target_tokens}'
        f'\t{max_memory_allocated_gb}\t{mean_memory_allocated_gb}'
        f'\t{max_memory_per_token_gb}\t{mean_memory_per_token_gb}'
        f'\t{torch.cuda.is_available()}'
        f'\t{total_source_tokens}\t{total_target_tokens}\t{total_all_tokens}'
        f'\t{peak_memory_no_model_gb}\t{peak_memory_model_only_gb}')

    with open(args.predictions_file, "w", encoding="utf-8") as f:
        for pred in preds:
            print(pred, file=f)

    # if torch.cuda.is_available():
    #     import matplotlib.pyplot as plt
    #
    #     plt.plot(num_tokens, [m / (1024 ** 3) for m in peak_sentence_memory], linestyle='None', marker='.')
    #     plt.show()
