#!/bin/bash

set -e -o pipefail -x

export srcd=w19edoc
export srctf=test.en.w19edoc
export srcpf=test.en.w19moendoc
export modelf="expm/w19edocpret/avg.h5"
export rsd=w19edoctrs
export rsf=$rsd/trans.txt
export pret_vcb=$tgtd/mdoc/src.vcb

export share_vcb=true

export cachedir=cache
export dataid=w19edocpret

export ngpu=1

export sort_decode=true
export debpe=true
export spm_bpe=false

export faext=".xz"

export tgtd=$cachedir/$dataid

export bpef=out.bpe

if $share_vcb; then
	export src_vcb=$tgtd/edoc/common.vcb
	export tgt_vcb=$src_vcb
else
	export src_vcb=$tgtd/edoc/src.vcb
	export tgt_vcb=$tgtd/edoc/tgt.vcb
fi

mkdir -p $rsd

if $sort_decode; then
	export srt_input_f=$tgtd/$srctf.srt$faext
	export srt_input_fp=$tgtd/$srcpf.srt$faext
	python tools/doc/sort.py $srcd/$srctf $srcd/$srcpf $srt_input_f $srt_input_fp 1048576
else
	export $srt_input_f=$srcd/$srctf
	export $srt_input_fp=$srcd/$srcpf
fi

python tools/doc/pret/mktest.py $srt_input_f $srt_input_fp $src_vcb $pret_vcb $tgtd/test.h5 $ngpu
python predict_doc_pret.py $tgtd/$bpef.srt $tgt_vcb $modelf

if $sort_decode; then
	python tools/restore.py $srcd/$srctf $srt_input_f $tgtd/$bpef.srt $tgtd/$bpef
	rm $srt_input_f $srt_input_fp $tgtd/$bpef.srt
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
