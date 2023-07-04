#!/bin/bash

set -e -o pipefail -x

export srcd=/home/yfguo/CCMT_model/bart_wright/transformer-edge/cache/16k_cache_For_multi
export srctf=src.dev.bpe.meng
export modelf="/home/yfguo/CCMT_model/bart_wright/transformer-edge/expm/rs/std/base_bart_multi_16k_without_encoder/step1/avg.h5"
export rsd=$srcd
export reference=tgt.dev.meng.zh

export tgt_vcb=/home/yfguo/CCMT_model/bart_wright/bart-base-chinese/vocab.txt

export cachedir=cache
export dataid=16k_multi2zh_meng_noencoder_trans

export ngpu=1

export sort_decode=true

export faext=".xz"

export tgtd=$cachedir/$dataid
export rsf=$tgtd/dec.test.txt
export trans=$tgtd/trans.txt

export src_vcb=/home/yfguo/CCMT_model/bart_wright/transformer-edge/cache/16k_cache_For_multi/rs1/srcvcb.txt
export bpef=out.bpe

mkdir -p $tgtd

#用map.py生成少数民族的id文件
export stif=$tgtd/$srctf.ids$faext
python /home/yfguo/CCMT_model/bart_wright/transformer-edge/tools/vocab/map.py $srcd/$srctf $src_vcb $stif &
wait

#用上方替代
#export stif=$tgtd/$srctf.ids$faext
#python tools/plm/map/bart.py $srcd/$srctf $src_vcb $stif

if $sort_decode; then
	export srt_input_f=$tgtd/$srctf.ids.srt$faext
	python tools/sort.py $stif $srt_input_f 1048576
else
	export srt_input_f=$stif
fi

python tools/plm/mktest.py $srt_input_f $tgtd/test.h5 $ngpu &
wait
python 16k_meng_multi_bart_predict.py $tgtd/$bpef $tgt_vcb $modelf &
wait

if $sort_decode; then
	python tools/restore.py $stif $srt_input_f $tgtd/$bpef $rsf
else
	mv $tgtd/$bpef $rsf
fi

python /home/yfguo/CCMT_model/bart_wright/transformer-edge/special_token_process.py $rsf $trans &
wait 
sacrebleu $srcd/$reference -i $trans -m bleu -b -w 4 --tokenize char --force