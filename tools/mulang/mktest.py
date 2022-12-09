#encoding: utf-8

import sys
from numpy import array as np_array, int32 as np_int32

from utils.fmt.mulang.single import batch_padder
from utils.fmt.vocab.token import ldvocab
from utils.h5serial import h5File

from cnfg.ihyp import *

# maxtoken should be the maxtoken in mkiodata.py / 2 / beam size roughly, similar for bsize

def handle(finput, fvocab_i, fvocab_task, frs, minbsize=1, expand_for_mulgpu=True, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, minfreq=False, vsize=False):
	vcbi, nwordi = ldvocab(fvocab_i, minf=minfreq, omit_vsize=vsize, vanilla=False)
	vcbtask, nwordtask = ldvocab(fvocab_task, minf=False, omit_vsize=False, vanilla=True)
	if expand_for_mulgpu:
		_bsize = bsize * minbsize
		_maxtoken = maxtoken * minbsize
	else:
		_bsize = bsize
		_maxtoken = maxtoken
	with h5File(frs, "w", libver=h5_libver) as rsf:
		src_grp = rsf.create_group("src")
		task_grp = rsf.create_group("task")
		curd = 0
		for i_d, taskd in batch_padder(finput, vcbi, vcbtask, _bsize, maxpad, maxpart, _maxtoken, minbsize):
			rid = np_array(i_d, dtype=np_int32)
			rtaskd = np_array(taskd, dtype=np_int32)
			wid = str(curd)
			src_grp.create_dataset(wid, data=rid, **h5datawargs)
			task_grp.create_dataset(wid, data=rtaskd, **h5datawargs)
			curd += 1
		rsf["ndata"] = np_array([curd], dtype=np_int32)
		rsf["nword"] = np_array([nwordi, nwordtask], dtype=np_int32)
	print("Number of batches: %d\nSource Vocabulary Size: %d\nNumber of Tasks: %d" % (curd, nwordi, nwordtask,))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
