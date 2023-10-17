# Sequence Shortening for Context-aware Machine Translation
Code for the paper "Sequence Shortening for Context-aware Machine Translation". 
Based on https://github.com/neulab/contextual-mt repository.

## Installation

Install:
- fairseq version 0.12.2 https://github.com/facebookresearch/fairseq#requirements-and-installation.
- requirements from ```requirements.txt``` file.
- COMET version 2.0.2 https://github.com/Unbabel/COMET
- sacreBLEU version 2.3.1 ```pip install sacrebleu==2.3.1```
- ContraPro contrastive dataset (En-De) https://github.com/ZurichNLP/ContraPro
- large contrastive dataset (En-Fr) https://github.com/rbawden/Large-contrastive-pronoun-testset-EN-FR



## Data preprocessing
Download the data for English-German and English-French language pairs from https://huggingface.co/datasets/iwslt2017/tree/c18a4f81a47ae6fa079fe9d32db288ddde38451d/data/2017-01-trnted/texts.
Extract it in the ```data/iwslt2017``` folder.

Update paths and variables in  ```scripts/data/iwslt2017/preprocess.sh``` and run it to preprocess the data.

## Training

In order to train the models use the following:
- Caching Tokens:
```shell
fairseq-train path/to/data/en-de/bin/ \
        --user-dir path/to/repo/contextual_mt \
        --task document_translation \
        --source-lang en \
        --target-lang de \
        --source-context-size 1 \
        --target-context-size 0 \
        --log-interval 10 \
        --arch caching_contextual_transformer \
        --share-decoder-input-output-embed \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --clip-norm 0.1 \
        --lr 5e-4 \
        --lr-scheduler inverse_sqrt \
        --warmup-updates 2500 \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --dropout 0.3 \
        --weight-decay 0.0001 \
        --max-tokens  4096 \
        --update-freq 8 \
        --seed 42 \
        --validate-interval 1 \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-remove-bpe sentencepiece \
        --eval-bleu-print-samples \
        --best-checkpoint-metric loss \
        --save-dir ./checkpoints \
        --no-epoch-checkpoints \
        --context-sentence-learned-pos \
        --separate-context-sentences \
        --patience 5
```
- Shortening - Pooling models (pooling type can be changed with ```--shortening-pooling-type``` argument with he possible values of ```"mean"```, ```"max"```, ```"linear"```):
```shell
fairseq-train path/to/data/en-de/bin/ \
        --user-dir path/to/repo/contextual_mt \
        --task document_translation \
        --source-lang en \
        --target-lang de \
        --source-context-size 1 \
        --target-context-size 0 \
        --log-interval 10 \
        --arch shortening_contextual_transformer \
        --share-decoder-input-output-embed \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --clip-norm 0.1 \
        --lr 5e-4 \
        --lr-scheduler inverse_sqrt \
        --warmup-updates 2500 \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --dropout 0.3 \
        --weight-decay 0.0001 \
        --max-tokens  4096 \
        --update-freq 8 \
        --seed 42 \
        --validate-interval 1 \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-remove-bpe sentencepiece \
        --eval-bleu-print-samples \
        --best-checkpoint-metric loss \
        --save-dir ./checkpoints \
        --no-epoch-checkpoints \
        --context-sentence-learned-pos \
        --shortening-use-pooling \
        --shortening-pooling-type "linear" \
        --shortening-groups 2 \
        --shortening-group-learned-pos \
        --separate-context-sentences \
        --patience 5
```

- Shortening - Grouping model:
```shell
fairseq-train path/to/data/en-de/bin/ \
        --user-dir path/to/repo/contextual_mt \
        --task document_translation \
        --source-lang en \
        --target-lang de \
        --source-context-size 1 \
        --target-context-size 0 \
        --log-interval 10 \
        --arch shortening_contextual_transformer \
        --share-decoder-input-output-embed \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --clip-norm 0.1 \
        --lr 5e-4 \
        --lr-scheduler inverse_sqrt \
        --warmup-updates 2500 \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --dropout 0.3 \
        --weight-decay 0.0001 \
        --max-tokens  4096 \
        --update-freq 8 \
        --seed 42 \
        --validate-interval 1 \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-remove-bpe sentencepiece \
        --eval-bleu-print-samples \
        --best-checkpoint-metric loss \
        --save-dir ./checkpoints \
        --no-epoch-checkpoints \
        --context-sentence-learned-pos \
        --shortening-groups 9 \
        --shortening-group-learned-pos \
        --shortening-ffn-hidden 512 \
        --shortening-use-sparsemax \
        --shortening-propagate-context-encoder-gradient \
        --shortening-propagate-context-size-gradient 2 \
        --separate-context-sentences \
        --patience 5
```

- Shortening - Selecting model:
```shell
fairseq-train path/to/data/en-de/bin/ \
        --user-dir path/to/repo/contextual_mt \
        --task document_translation \
        --source-lang en \
        --target-lang de \
        --source-context-size 1 \
        --target-context-size 0 \
        --log-interval 10 \
        --arch shortening_contextual_transformer \
        --share-decoder-input-output-embed \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --clip-norm 0.1 \
        --lr 5e-4 \
        --lr-scheduler inverse_sqrt \
        --warmup-updates 2500 \
        --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --dropout 0.3 \
        --weight-decay 0.0001 \
        --max-tokens  4096 \
        --update-freq 8 \
        --seed 42 \
        --validate-interval 1 \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-remove-bpe sentencepiece \
        --eval-bleu-print-samples \
        --best-checkpoint-metric loss \
        --save-dir ./checkpoints \
        --no-epoch-checkpoints \
        --context-sentence-learned-pos \
        --shortening-groups 10 \
        --shortening-group-learned-pos \
        --shortening-ffn-hidden 512 \
        --shortening-use-sparsemax \
        --shortening-selecting-groups \
        --shortening-propagate-context-encoder-gradient \
        --shortening-propagate-context-size-gradient 1 \
        --separate-context-sentences \
        --patience 5
```

## Inference

To translate the test subset:
```shell
PYTHONPATH=path/to/repo python -m contextual_mt.docmt_translate \ 
    --path checkpoints/ \
    --source-lang en \
    --target-lang de \
    --source-file path/to/data/en-de/test.en-de.en \
    --predictions-file preds.test.en-de.de \
    --docids-file path/to/data/en-de/test.en-de.docids \
    --beam 5 \
    --split-name test \
    --separate-context-sentences
```

## Evaluation

To calculate BLEU score:
```shell
sacrebleu path/to/data/en-de/test.en-de.de \ 
    -i preds.test.en-de.de \
    -m bleu chrf ter -w 4
```

To calculate COMET score:
```shell
comet-score -s path/to/data/en-de/test.en-de.en \ 
    -t preds.test.en-de.de \
    -r path/to/data/en-de/test.en-de.de \
    --model Unbabel/wmt22-comet-da \
    --quiet --only_system
```

Generate ContraPro scores:
```shell
PYTHONPATH=path/to/repo python -m contextual_mt.docmt_contrastive_eval \
    --source-file path/to/contrapro/contrapro.text.en \
    --src-context-file path/to/contrapro/contrapro.context.en \
    --target-file path/to/contrapro/contrapro.text.de \
    --tgt-context-file path/to/contrapro/contrapro.context.de \
    --source-context-size 1 \
    --target-context-size 0 \
    --source-lang en \
    --target-lang de \
    --dataset contrapro \
    --path checkpoints/ \
    --save-scores contrapro.scores \
    --preds-file preds.contrapro.de \
    --separate-context-sentences
```

Calculate ContraPro accuracy:
```shell
python path/to/contrapro/evaluate.py \
    --reference path/to/contrapro/contrapro.json \
    --scores contrapro.scores \
    --maximize
```