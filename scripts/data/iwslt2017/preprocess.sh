#!/bin/bash

set -euo pipefail

# Replace appropriate variables for paths, languages and configs

REPO=.

# Download and extract the dataset from
# https://huggingface.co/datasets/iwslt2017/tree/c18a4f81a47ae6fa079fe9d32db288ddde38451d/data/2017-01-trnted/texts

data="./data/iwslt2017/en-fr"
raw=$data/raw
prep=$data/prep
bin=$data/bin
src_l="en"
tgt_l="fr"#"de"
vocab_size=20000
joint_vocab=false

mkdir -p $raw $prep $bin

#archive=$raw/${url##*/}
#echo $archive
#if [ -f "$archive" ]; then
#    echo "$archive already exists, skipping download and extraction..."
#else
#    wget -P $raw $url
#    if [ -f "$archive" ]; then
#        echo "$url successfully downloaded."
#    else
#        echo "$url not successfully downloaded."
#        exit 1
#    fi
#
#    tar --strip-components=1 -C $raw -xzvf $archive
#fi
#
#echo "extract from raw data..."
#rm -f $data/*.${src_l}-${tgt_l}.*
python ${REPO}/scripts/data/iwslt2017/prepare_corpus.py $raw $data -s $src_l -t $tgt_l 

echo "building sentencepiece model..."
if [ "$joint_vocab" = true ]; then
    cat $data/train.${src_l}-${tgt_l}.${src_l} $data/train.${src_l}-${tgt_l}.${tgt_l} > /tmp/train.${src_l}-${tgt_l}.all
    python scripts/spm_train.py /tmp/train.${src_l}-${tgt_l}.all \
        --model-prefix $prep/spm \
        --vocab-file $prep/dict.${src_l}-${tgt_l}.txt \
        --vocab-size $vocab_size
    rm /tmp/train.${src_l}-${tgt_l}.all
    ln -s $prep/dict.${src_l}-${tgt_l}.txt $prep/dict.${src_l}.txt
    ln -s $prep/dict.${src_l}-${tgt_l}.txt $prep/dict.${tgt_l}.txt
    ln -s $prep/spm.model $prep/spm.${src_l}.model
    ln -s $prep/spm.model $prep/spm.${tgt_l}.model
else
    for lang in $src_l $tgt_l; do
        python scripts/spm_train.py $data/train.${src_l}-${tgt_l}.${lang} \
            --model-prefix $prep/spm.${lang} \
            --vocab-file $prep/dict.${lang}.txt \
            --vocab-size $vocab_size
    done
fi


echo "applying sentencepiece model..."
for split in "train" "valid" "test"; do 
    for lang in $src_l $tgt_l; do 
        python scripts/spm_encode.py \
            --model $prep/spm.$lang.model \
                < $data/${split}.${src_l}-${tgt_l}.${lang} \
                > $prep/${split}.${src_l}-${tgt_l}.${lang}
    done
done

echo "binarizing..."
fairseq-preprocess \
    --source-lang ${src_l} --target-lang ${tgt_l} \
    --trainpref ${prep}/train.${src_l}-${tgt_l} --validpref ${prep}/valid.${src_l}-${tgt_l} --testpref ${prep}/test.${src_l}-${tgt_l} \
    --srcdict ${prep}/dict.${src_l}.txt --tgtdict ${prep}/dict.${tgt_l}.txt \
    --destdir ${bin} \
    --workers 20

cp $data/*.docids $bin