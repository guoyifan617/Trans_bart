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

export faext=".xz"

export wkd=$cachedir/$dataid

mkdir -p $wkd

export stif=$wkd/src.train.ids$faext
export sdif=$wkd/src.dev.ids$faext
if $do_map; then
	python tools/plm/map/custbert.py $srcd/$srctf $src_vcb $stif &
	python tools/plm/map/custbert.py $srcd/$srcvf $src_vcb $sdif &
	wait
fi

export stsf=$wkd/src.train.srt$faext
export ttsf=$wkd/tgt.train.srt$faext
export sdsf=$wkd/src.dev.srt$faext
export tdsf=$wkd/tgt.dev.srt$faext
if $do_sort; then
	python tools/sort.py $stif $srcd/$tgttf $stsf $ttsf $maxtokens &
	# use the following command to sort a very large dataset with limited memory
	#bash tools/lsort/sort.sh $stif $srcd/$tgttf $stsf $ttsf $maxtokens &
	python tools/sort.py $sdif $srcd/$tgtvf $sdsf $tdsf 1048576 &
	wait
fi

python tools/plm/mkiodata.py $stsf $ttsf $wkd/$rsf_train $ngpu &
python tools/plm/mkiodata.py $sdsf $tdsf $wkd/$rsf_dev $ngpu &
wait
