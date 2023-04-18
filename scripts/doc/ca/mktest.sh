#!/bin/bash

set -e -o pipefail -x

export srcd=w19edoc
export srctf=test.en.w19edoc
export tgttf=test.base_avg.bpe
export modelf="expm/w19edoc/doc/ca/checkpoint.h5"
export rsd=w19edoctrs
export rsf=$rsd/trans.txt

export share_vcb=true

export cachedir=cache
export dataid=w19edoc

export ngpu=1

export sort_decode=true
export debpe=true
export spm_bpe=false

export faext=".xz"

export tgtd=$cachedir/$dataid

export bpef=out.bpe

if $share_vcb; then
	export src_vcb=$tgtd/common.vcb
	export tgt_vcb=$src_vcb
else
	export src_vcb=$tgtd/src.vcb
	export tgt_vcb=$tgtd/tgt.vcb
fi

mkdir -p $rsd

if $sort_decode; then
	export srt_input_f=$tgtd/$srctf.srt$faext
	export srt_input_ft=$tgtd/$tgttf.srt$faext
	python tools/doc/sort.py $srcd/$srctf $srcd/$tgttf $srt_input_f $srt_input_ft 1048576
else
	export srt_input_f=$srcd/$srctf
	export srt_input_ft=$srcd/$tgttf
fi

python tools/doc/para/mkiodata.py $srt_input_f $srt_input_ft $src_vcb $tgt_vcb $tgtd/test.h5 $ngpu
python predict_doc_ca.py $tgtd/$bpef.srt $tgt_vcb $modelf

if $sort_decode; then
	python tools/doc/para/restore.py $srcd/$srctf $srcd/$srctf $srcd/$tgttf $srt_input_f $tgtd/$bpef.srt $tgtd/$bpef
	rm $srt_input_f $srt_input_ft $tgtd/$bpef.srt
else
	mv $tgtd/$bpef.srt $tgtd/$bpef
fi

if $debpe; then
	if $spm_bpe; then
		python tools/spm/decode.py --model $tgtd/bpe.model --input_format piece --input $tgtd/$bpef > $rsf

	else
		sed -r 's/(@@ )|(@@ ?$)//g' < $tgtd/$bpef > $rsf
	fi
	rm $tgtd/$bpef
else
	mv $tgtd/$bpef $rsf
fi
rm $tgtd/test.h5
