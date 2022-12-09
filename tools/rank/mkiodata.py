#encoding: utf-8

import sys
from numpy import array as np_array, int32 as np_int32

from utils.fmt.triple import batch_padder
from utils.fmt.vocab.token import ldvocab
from utils.h5serial import h5File

from cnfg.ihyp import *

def handle(finput, fref, ftarget, fvocab, frs, minbsize=1, expand_for_mulgpu=True, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, minfreq=False, vsize=False):
	vcb, nword = ldvocab(fvocab, minfreq, vsize)
	if expand_for_mulgpu:
		_bsize = bsize * minbsize
		_maxtoken = maxtoken * minbsize
	else:
		_bsize = bsize
		_maxtoken = maxtoken
	with h5File(frs, "w", libver=h5_libver) as rsf:
		src_grp = rsf.create_group("src")
		ref_grp = rsf.create_group("ref")
		tgt_grp = rsf.create_group("tgt")
		curd = 0
		for i_d, rd, td in batch_padder(finput, fref, ftarget, vcb, _bsize, maxpad, maxpart, _maxtoken, minbsize):
			rid = np_array(i_d, dtype=np_int32)
			rrd = np_array(rd, dtype=np_int32)
			rtd = np_array(td, dtype = numpy.float32)
			wid = str(curd)
			src_grp.create_dataset(wid, data=rid, **h5datawargs)
			ref_grp.create_dataset(wid, data=rrd, **h5datawargs)
			tgt_grp.create_dataset(wid, data=rtd, **h5datawargs)
			curd += 1
		rsf["ndata"] = np_array([curd], dtype=np_int32)
		rsf["nword"] = np_array([nword], dtype=np_int32)
	print("Number of batches: %d\nVocabulary Size: %d" % (curd, nword,))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]))
