#encoding: utf-8

import sys
from numpy import array as np_array, int32 as np_int32, int8 as np_int8

from utils.fmt.gec.gector.triple import batch_padder
from utils.h5serial import h5File

from cnfg.ihyp import *

def handle(finput, fedit, ftarget, frs, minbsize=1, expand_for_mulgpu=True, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, minfreq=False, vsize=False):

	if expand_for_mulgpu:
		_bsize = bsize * minbsize
		_maxtoken = maxtoken * minbsize
	else:
		_bsize = bsize
		_maxtoken = maxtoken
	with h5File(frs, "w", libver=h5_libver) as rsf:
		src_grp = rsf.create_group("src")
		edt_grp = rsf.create_group("edt")
		tgt_grp = rsf.create_group("tgt")
		curd = 0
		for i_d, ed, td in batch_padder(finput, fedit, ftarget, _bsize, maxpad, maxpart, _maxtoken, minbsize):
			rid = np_array(i_d, dtype=np_int32)
			red = np_array(ed, dtype=np_int8)
			rtd = np_array(td, dtype=np_int32)
			wid = str(curd)
			src_grp.create_dataset(wid, data=rid, **h5datawargs)
			edt_grp.create_dataset(wid, data=red, **h5datawargs)
			tgt_grp.create_dataset(wid, data=rtd, **h5datawargs)
			curd += 1
		rsf["ndata"] = np_array([curd], dtype=np_int32)
	print("Number of batches: %d" % curd)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
