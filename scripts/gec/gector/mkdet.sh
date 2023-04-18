#!/bin/bash

set -e -o pipefail -x

export cachedir=cache
export dataid=gecdet/parallel

export srcd=$cachedir/$dataid
export srctf=src.train.txt
export tgttf=tgt.train.ids
export srcvf=src.dev.txt
export tgtvf=tgt.dev.ids
export src_vcb=~/plm/custbert/char.vcb

export rsf_train=train.h5
export rsf_dev=dev.h5

export maxtokens=512

export ngpu=1

export do_map=true
export do_sort=true

export wkd=$cachedir/$dataid

mkdir -p $wkd

if $do_map; then
	python tools/plm/map/custbert.py $srcd/$srctf $src_vcb $wkd/src.train.ids &
	python tools/plm/map/custbert.py $srcd/$srcvf $src_vcb $wkd/src.dev.ids &
	wait
fi

if $do_sort; then
	python tools/sort.py $wkd/src.train.ids $srcd/$tgttf $wkd/src.train.srt $wkd/tgt.train.srt $maxtokens &
	# use the following command to sort a very large dataset with limited memory
	#bash tools/lsort/sort.sh $wkd/src.train.ids $srcd/$tgttf $wkd/src.train.srt $wkd/tgt.train.srt $maxtokens &
	python tools/sort.py $wkd/src.dev.ids $srcd/$tgtvf $wkd/src.dev.srt $wkd/tgt.dev.srt 1048576 &
	wait
fi

python tools/plm/mkiodata.py $wkd/src.train.srt $wkd/tgt.train.srt $wkd/$rsf_train $ngpu &
python tools/plm/mkiodata.py $wkd/src.dev.srt $wkd/tgt.dev.srt $wkd/$rsf_dev $ngpu &
wait
