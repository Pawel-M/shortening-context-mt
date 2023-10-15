# From: https://github.com/neulab/contextual-mt

import argparse
import os
import json
import pickle

import torchinfo
import tqdm

import torch

from fairseq import utils, hub_utils
from fairseq.sequence_scorer import SequenceScorer

import sentencepiece as sp

import contextual_mt  # noqa: F401
from contextual_mt import DocumentTranslationTask, ContextualSequenceGenerator
from contextual_mt.contextual_dataset import collate as contextual_collate
from contextual_mt.contextual_list_dataset import collate as contextual_list_collate
from fairseq.data.language_pair_dataset import collate as raw_collate
from contextual_mt.utils import create_context, create_context_list, encode, decode


def load_contrastive(
        source_file,
        target_file,
        src_context_file,
        tgt_context_file,
        dataset="contrapro",
):
    """
    reads the contrapro (or large_contrastive) or bawden contrastive dataset
    """
    if dataset == "contrapro":
        pronouns = ("Er", "Sie", "Es")
    elif dataset in ("bawden", "large_contrastive"):
        pronouns = None
    else:
        raise ValueError("should not get here")

    # load files needed
    # and binarize
    with open(source_file, "r") as src_f, open(target_file, "r") as tgt_f, open(
            src_context_file
    ) as src_ctx_f, open(tgt_context_file) as tgt_ctx_f:
        srcs = []
        srcs_context = []
        tgts_context = []
        all_tgts = []
        tgt_labels = []
        src_lines = src_f.readlines()
        src_ctx_lines = src_ctx_f.readlines()
        tgt_lines = tgt_f.readlines()
        tgt_ctx_lines = tgt_ctx_f.readlines()
        assert len(src_lines) == len(
            tgt_lines
        ), "source and target files have different sizes"
        assert len(src_ctx_lines) == len(
            tgt_ctx_lines
        ), "src_content and tgt_context files have different_sizes"
        assert (
                len(src_ctx_lines) % len(src_lines) == 0
        ), "src_context file lines aren't multiple of source lines"
        included_context_size = len(src_ctx_lines) // len(src_lines)

        index = 0
        while index < len(src_lines):
            i = 0
            src = None
            tgts = []
            while (index + i) < len(src_lines) and (
                    (
                            dataset in ("contrapro", "large_contrastive")
                            and (src is None or src == src_lines[index + i])
                    )
                    or (dataset == "bawden" and (i < 2))
            ):
                src = src_lines[index + i]
                tgt = tgt_lines[index + i]
                src_context = [
                    src_ctx_lines[(index + i) * included_context_size + j].strip()
                    for j in range(included_context_size)
                ]
                tgt_context = [
                    tgt_ctx_lines[(index + i) * included_context_size + j].strip()
                    for j in range(included_context_size)
                ]
                tgts.append(tgt.strip())
                i += 1

            lower_gold = tgts[0].lower()
            tokenized_gold = lower_gold.split(" ")
            # if for some reason simple tokenization
            # doesn't work, just count the pron that
            # appears more times
            max_count, best_pron = 0, None
            if pronouns is not None:
                for pron in pronouns:
                    if pron.lower() in tokenized_gold:
                        best_pron = pron
                        max_count = 1
                        break
                    count = lower_gold.count(pron.lower())
                    if count > max_count:
                        best_pron = pron
                        max_count = count
                if max_count == 0:
                    raise ValueError(
                        f"no pronoun found in one of the sentences: {tgts[0]}"
                    )
            else:
                best_pron = None

            tgt_labels.append(best_pron)

            srcs.append(src.strip())
            all_tgts.append(tgts)
            srcs_context.append(src_context)
            tgts_context.append(tgt_context)
            index += i

    assert len([t for tgt in all_tgts for t in tgt]) == len(
        tgt_lines
    ), "ended up with differnt number of lines..."
    return srcs, all_tgts, tgt_labels, srcs_context, tgts_context


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--src-context-file", required=True)
    parser.add_argument("--target-file", required=True)
    parser.add_argument("--tgt-context-file", required=True)
    parser.add_argument("--source-context-size", default=0, type=int)
    parser.add_argument("--target-context-size", default=0, type=int)
    parser.add_argument("--source-lang", default=None)
    parser.add_argument("--target-lang", default=None)
    parser.add_argument(
        "--dataset", choices=("contrapro", "bawden", "large_contrastive"), default="contrapro"
    )
    parser.add_argument(
        "--path", required=True, metavar="FILE", help="path to model file"
    )
    parser.add_argument("--checkpoint-file", default="checkpoint_best.pt")
    parser.add_argument("--checkpoint-prefix", default="best")
    parser.add_argument("--save-scores", default=None, type=str)
    parser.add_argument("--preds-file", default=None, type=str)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=("number of sentences to inference in parallel"),
    )
    parser.add_argument("--no-progress", default=False, action="store_true", help="do not show progress bar", )
    parser.add_argument("--limit-size", default=None, type=int, help="limit the size of the dataset", )
    parser.add_argument("--separate-context-sentences", default=False, action="store_true",
                        help="if set, separates context sentences into a list", )
    args = parser.parse_args()
    return args


def get_args():
    datasets_path = '../../Datasets/discourse-mt-test-sets/test-sets/'
    model_path = '../results/iwslt17_en-fr_fast/coword/'
    args = argparse.Namespace()
    args.source_file = datasets_path + 'anaphora.current.en'
    args.src_context_file = datasets_path + 'anaphora.prev.en'
    args.target_file = datasets_path + 'anaphora.current.fr'
    args.tgt_context_file = datasets_path + 'anaphora.prev.fr'
    args.source_context_size = 1
    args.target_context_size = 0
    args.source_lang = 'en'
    args.target_lang = 'fr'
    args.dataset = 'bawden'
    args.path = model_path + 'checkpoints'
    args.checkpoint_file = 'checkpoint_last.pt'
    args.checkpoint_prefix = 'last'
    args.save_scores = model_path + 'bawden.scores'
    args.batch_size = 8
    args.no_progress = False
    args.separate_context_sentences = False
    return args


def load_model(path, checkpoint_file, source_context_size, target_context_size):
    # load pretrained model, set eval and send to cuda
    pretrained = hub_utils.from_pretrained(
        path, checkpoint_file=checkpoint_file
    )
    models = pretrained["models"]
    for model in models:
        if torch.cuda.is_available():
            model.cuda()
        model.eval()

    # load dict, params and generator from task
    src_dict = pretrained["task"].src_dict
    tgt_dict = pretrained["task"].tgt_dict
    if isinstance(pretrained["task"], DocumentTranslationTask):
        concat_model = False
        source_context_size = pretrained["task"].args.source_context_size
        target_context_size = pretrained["task"].args.target_context_size
    else:
        concat_model = True
        source_context_size = source_context_size
        target_context_size = target_context_size

    return pretrained, models, src_dict, tgt_dict, concat_model, source_context_size, target_context_size


def load_spm(args):
    # load sentencepiece models (assume they are in the checkpoint dirs)
    if os.path.exists(os.path.join(args.path, "spm.model")):
        spm = sp.SentencePieceProcessor()
        spm.Load(os.path.join(args.path, "spm.model"))
        src_spm = spm
        tgt_spm = spm
    else:
        src_spm = sp.SentencePieceProcessor()
        src_spm.Load(os.path.join(args.path, f"spm.{args.source_lang}.model"))
        tgt_spm = sp.SentencePieceProcessor()
        tgt_spm.Load(os.path.join(args.path, f"spm.{args.target_lang}.model"))
    return src_spm, tgt_spm


def binarize(sources, srcs_contexts, all_tgts, tgts_contexts, src_spm, src_dict, tgt_spm, tgt_dict):
    srcs = []
    src_tokens = []
    for s in sources:
        e, t = encode(s, src_spm, src_dict, return_tokenized=True)
        srcs.append(e)
        src_tokens.append(t)
    # srcs = [encode(s, src_spm, src_dict) for s in srcs]
    all_tgts = [[encode(s, tgt_spm, tgt_dict) for s in tgts] for tgts in all_tgts]
    src_ctxs = []
    src_ctx_tokens = []
    for context in srcs_contexts:
        ctx = []
        for s in context:
            e, t = encode(s, src_spm, src_dict, return_tokenized=True)
            ctx.append(e)
            src_ctx_tokens.append(t)
        src_ctxs.append(ctx)
    # srcs_context = [
    #     [encode(s, src_spm, src_dict) for s in context] for context in srcs_contexts
    # ]
    tgts_context = [
        [encode(s, tgt_spm, tgt_dict) for s in context] for context in tgts_contexts
    ]

    return srcs, all_tgts, src_ctxs, tgts_context, src_tokens, src_ctx_tokens


def main():
    args = parse_args()
    # args = get_args()

    pretrained, models, src_dict, tgt_dict, concat_model, source_context_size, target_context_size = load_model(
        args.path,
        args.checkpoint_file,
        args.source_context_size,
        args.target_context_size
    )

    # generator = pretrained["task"].build_generator(
    #     models, args, seq_gen_cls=ContextualSequenceGenerator
    # )

    separate_context_sentences = (
        pretrained["task"].args.separate_context_sentences
        if args.separate_context_sentences is None
        else args.separate_context_sentences
    )
    print('separate_context_sentences', separate_context_sentences)

    torchinfo.summary(models[0])

    scorer = SequenceScorer(tgt_dict)
    src_spm, tgt_spm = load_spm(args)
    # load files
    srcs, all_tgts, tgt_labels, srcs_contexts, tgts_contexts = load_contrastive(
        args.source_file,
        args.target_file,
        args.src_context_file,
        args.tgt_context_file,
        dataset=args.dataset,
    )
    srcs, all_tgts, srcs_context, tgts_context, source_tokens, src_ctx_tokens = binarize(
        srcs,
        srcs_contexts,
        all_tgts,
        tgts_contexts,
        src_spm,
        src_dict,
        tgt_spm,
        tgt_dict
    )
    label_corrects = {label: [] for label in set(tgt_labels)}
    dataset_size = sum(1 for _ in srcs)
    if args.limit_size is not None:
        dataset_size = min(dataset_size, args.limit_size)
    bar = tqdm.tqdm(total=dataset_size) if not args.no_progress else None
    corrects = []
    attentions, src_log, src_context_log, tgt_context_log, tgt_log = [], [], [], [], []
    all_scores = []
    all_samples = []
    preds = []
    scores = []
    groups = []
    ctx_groups = []
    generated_tokens = []
    # generator = pretrained["task"].build_generator(
    #     models, args, seq_gen_cls=ContextualSequenceGenerator
    # )
    for i, (src, src_ctx, contr_tgts, tgt_ctx, label) in enumerate(zip(
            srcs, srcs_context, all_tgts, tgts_context, tgt_labels
    )):
        samples = []
        for tgt in contr_tgts:
            if concat_model:
                if separate_context_sentences:
                    src_ctx_tensor = create_context_list(
                        src_ctx, source_context_size, src_dict.index("<brk>")
                    )
                    tgt_ctx_tensor = create_context_list(
                        tgt_ctx, target_context_size, tgt_dict.index("<brk>")
                    )
                else:
                    src_ctx_tensor = create_context(
                        src_ctx, source_context_size, src_dict.index("<brk>")
                    )
                    tgt_ctx_tensor = create_context(
                        tgt_ctx, target_context_size, tgt_dict.index("<brk>")
                    )
                    if len(src_ctx_tensor) > 0:
                        src_ctx_tensor = torch.cat(
                            [src_ctx_tensor, torch.tensor([src_dict.index("<brk>")])]
                        )

                    if len(tgt_ctx_tensor) > 0:
                        tgt_ctx_tensor = torch.cat(
                            [tgt_ctx_tensor, torch.tensor([tgt_dict.index("<brk>")])]
                        )
                full_src = torch.cat(
                    [src_ctx_tensor, src, torch.tensor([src_dict.eos()])]
                )

                full_tgt = torch.cat(
                    [tgt_ctx_tensor, tgt, torch.tensor([tgt_dict.eos()])]
                )
                sample = {"id": 0, "source": full_src, "target": full_tgt}
            else:
                if separate_context_sentences:
                    src_ctx_tensor = create_context_list(
                        src_ctx,
                        source_context_size,
                        src_dict.index("<brk>"),
                        src_dict.eos(),
                    )
                    tgt_ctx_tensor = create_context_list(
                        tgt_ctx,
                        target_context_size,
                        tgt_dict.index("<brk>"),
                        tgt_dict.eos(),
                    )
                else:
                    src_ctx_tensor = create_context(
                        src_ctx,
                        source_context_size,
                        src_dict.index("<brk>"),
                        src_dict.eos(),
                    )
                    tgt_ctx_tensor = create_context(
                        tgt_ctx,
                        target_context_size,
                        tgt_dict.index("<brk>"),
                        tgt_dict.eos(),
                    )

                full_src = torch.cat([src, torch.tensor([src_dict.eos()])])
                full_tgt = torch.cat([tgt, torch.tensor([tgt_dict.eos()])])
                sample = {
                    "id": 0,
                    "source": full_src,
                    "src_context": src_ctx_tensor,
                    "target": full_tgt,
                    "tgt_context": tgt_ctx_tensor,
                }
            samples.append(sample)

        if concat_model:
            sample = raw_collate(
                samples, pad_idx=src_dict.pad(), eos_idx=src_dict.eos()
            )
        else:
            if separate_context_sentences:
                sample = contextual_list_collate(
                    samples,
                    pad_id=src_dict.pad(),
                    eos_id=src_dict.eos(),
                )
            else:
                sample = contextual_collate(
                    samples,
                    pad_id=src_dict.pad(),
                    eos_id=src_dict.eos(),
                )
        if torch.cuda.is_available():
            sample = utils.move_to_cuda(sample)

        # output = pretrained["task"].inference_step(generator, models, sample)
        # print(sample['net_input'])
        net_input = {
            'src_tokens': sample['net_input']['src_tokens'],
            'src_lengths': sample['net_input']['src_lengths'],
            'src_ctx_tokens': sample['net_input']['src_ctx_tokens'],
            'src_ctx_lengths': sample['net_input']['src_ctx_lengths'],
        }
        enc_out = models[0].encoder.forward_torchscript(net_input)

        # for batch_idx in range(len(samples)):
        # decode hypothesis
        # hyp_ids = output[batch_idx][0]["tokens"].cpu()
        # preds.append(decode(hyp_ids, tgt_spm, tgt_dict))
        # collect other information
        if 'groups' in enc_out:
            enc_groups = enc_out['groups']
            if len(enc_groups) > 0:
                current_groups = enc_groups[-1]
                gs = current_groups[0].detach().cpu()
                groups.append(gs)
                for g in enc_groups[0:-1]:
                    ctx_groups.append(g[0].detach().cpu())

            # scores.append(output[batch_idx][0]["positional_scores"].cpu().tolist())

            # collect output to be prefix for next utterance
            # idx = batch_map[batch_idx]
            # if args.gold_target_context:
            #     tgt_context_lines[idx].append(batch_targets[batch_idx])
            # else:
            #     tgt_context_lines[idx].append(
            #         hyp_ids[:-1] if hyp_ids[-1] == tgt_dict.eos() else hyp_ids
            #     )
        hyps = scorer.generate(models, sample)
        scores = [h[0]["score"] for h in hyps]
        all_scores = all_scores + scores
        all_samples = all_samples + samples

        most_likely = torch.argmax(torch.stack(scores))
        correct = most_likely == 0
        corrects.append(correct)
        label_corrects[label].append(correct)

        # save info for attention visualization
        hyp_ids = hyps[most_likely][0]["tokens"].cpu()
        d, t = decode(hyp_ids, tgt_spm, tgt_dict, return_tokenized=True)
        generated_tokens.append(t)

        attentions.append(hyps[most_likely][0]["attention"].cpu())
        src_log.append(src_dict.string(samples[most_likely]["source"]) + " <eos>")
        tgt_log.append("<eos> " + tgt_dict.string(samples[most_likely]["target"]))
        if separate_context_sentences:
            src_context_log.append(
                [src_dict.string(c) + " <eos>" for c in samples[most_likely]["src_context"]]
            )
            tgt_context_log.append(
                ["<eos> " + tgt_dict.string(c) for c in samples[most_likely]["tgt_context"]]
            )
        else:
            src_context_log.append(
                src_dict.string(samples[most_likely]["src_context"]) + " <eos>"
            )
            tgt_context_log.append(
                "<eos> " + tgt_dict.string(samples[most_likely]["tgt_context"])
            )

        if bar is not None:
            bar.update(1)

        if args.limit_size is not None and i + 1 >= args.limit_size:
            break

    if bar is not None:
        bar.close()

    if None not in label_corrects:
        print("Pronoun accs...")
        for label, l_corrects in label_corrects.items():
            print(f" {label}: {torch.stack(l_corrects).float().mean().item()}")
    print(f"Total Acc: {torch.stack(corrects).float().mean().item()}")

    print("Saving info")
    with open("log.json", "w") as f:
        for src, src_context, tgt, tgt_context, attention, correct in zip(
                src_log, src_context_log, tgt_log, tgt_context_log, attentions, corrects
        ):
            d = json.dumps(
                {
                    "correct": correct.item(),
                    "source": src,
                    "source_context": src_context,
                    "target": tgt,
                    "target_context": tgt_context,
                    "attention": attention.tolist(),
                }
            )
            print(d, file=f)

    if args.save_scores is not None:
        with open(args.save_scores, "w") as f:
            for score in all_scores:
                print(score.item(), file=f)

    if args.preds_file is not None:
        with open(args.preds_file, "w", encoding="utf-8") as f:
            for pred in preds:
                print(pred, file=f)

    base_name = args.dataset + '.' + args.checkpoint_prefix + '.'
    attentions_file = base_name + 'attentions.pt'
    torch.save([a.transpose(0, 1) for a in attentions], attentions_file)

    if len(groups) > 0:
        groups_file = base_name + 'groups.pt'
        tokens_file = base_name + 'source_tokens.pkl'
        generated_tokens_file = base_name + 'generated_tokens.pkl'
        print(f'Saving groups to "{groups_file}" file and tokens to "{tokens_file}" file.')
        torch.save(groups, groups_file)
        with open(tokens_file, 'wb') as f:
            pickle.dump(source_tokens, f)
        with open(generated_tokens_file, 'wb') as f:
            pickle.dump(generated_tokens, f)

        ctx_groups_file = base_name + 'ctx_groups.pt'
        ctx_tokens_file = base_name + 'ctx_tokens.pkl'
        print(f'Saving context groups to "{groups_file}" file and context tokens to "{tokens_file}" file.')
        torch.save(ctx_groups, ctx_groups_file)
        with open(ctx_tokens_file, 'wb') as f:
            pickle.dump(src_ctx_tokens, f)

    return (
        label_corrects,
        corrects,
        attentions,
        src_log,
        src_context_log,
        tgt_context_log,
        tgt_log,
        all_scores,
        all_samples,
        preds,
        scores
    )


if __name__ == "__main__":
    (
        label_corrects,
        corrects,
        attentions,
        src_log,
        src_context_log,
        tgt_context_log,
        tgt_log,
        all_scores,
        all_samples,
        preds,
        scores
    ) = main()
