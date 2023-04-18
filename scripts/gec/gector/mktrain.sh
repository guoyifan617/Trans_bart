#!/bin/bash

set -e -o pipefail -x

export cachedir=cache
export dataid=gector

export srcd=$cachedir/$dataid
export srctf=src.train.txt
export tgttf=tgt.train.txt
export srcvf=src.dev.txt
export tgtvf=tgt.dev.txt
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

export stif=$wkd/$srctf.ids$faext
export ttif=$wkd/$tgttf.ids$faext
export sdif=$wkd/$srcvf.ids$faext
export tdif=$wkd/$tgtvf.ids$faext
export stcf=$wkd/src.train.ids$faext
export etcf=$wkd/edit.train.ids$faext
export ttcf=$wkd/tgt.train.ids$faext
export sdcf=$wkd/src.dev.ids$faext
export edcf=$wkd/edit.dev.ids$faext
export tdcf=$wkd/tgt.dev.ids$faext
if $do_map; then
	python tools/plm/map/custbert.py $srcd/$srctf $src_vcb $stif &
	python tools/plm/map/custbert.py $srcd/$tgttf $src_vcb $ttif &
	python tools/plm/map/custbert.py $srcd/$srcvf $src_vcb $sdif &
	python tools/plm/map/custbert.py $srcd/$tgtvf $src_vcb $tdif &
	wait
	python tools/gec/gector/convert.py $stif $ttif $stcf $etcf $ttcf &
	python tools/gec/gector/convert.py $sdif $tdif $sdcf $edcf $tdcf &
	wait
fi

export stsf=$wkd/src.train.srt$faext
export etsf=$wkd/edit.train.srt$faext
export ttsf=$wkd/tgt.train.srt$faext
export sdsf=$wkd/src.dev.srt$faext
export edsf=$wkd/edit.dev.srt$faext
export tdsf=$wkd/tgt.dev.srt$faext
if $do_sort; then
	python tools/sort.py $stcf $etcf $ttcf $stsf $etsf $ttsf $maxtokens &
	# use the following command to sort a very large dataset with limited memory
	#bash tools/lsort/sort.sh $stcf $etcf $ttcf $stsf $etsf $ttsf $maxtokens &
	python tools/sort.py $sdcf $edcf $tdcf $sdsf $edsf $tdsf 1048576 &
	wait
fi

python tools/gec/gector/mkiodata.py $stsf $etsf $ttsf $wkd/$rsf_train $ngpu &
python tools/gec/gector/mkiodata.py $sdsf $edsf $tdsf $wkd/$rsf_dev $ngpu &
wait
