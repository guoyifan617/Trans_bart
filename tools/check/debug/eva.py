#encoding: utf-8

import sys
import torch

from transformer.NMT import NMT
from utils.fmt.vocab.base import reverse_dict
from utils.fmt.vocab.token import ldvocab
from utils.h5serial import h5File, h5load
from utils.tqdm import tqdm

import cnfg.base as cnfg
from cnfg.ihyp import *

def eva(ed, i, model):

	src_grp, tgt_grp = ed["src"], ed["tgt"]
	bid = str(i)
	seq_batch = torch.from_numpy(src_grp[bid][()])
	seq_o = torch.from_numpy(tgt_grp[bid][()])
	lo = seq_o.size(1) - 1
	if cuda_device:
		seq_batch = seq_batch.to(cuda_device, non_blocking=True)
		seq_o = seq_o.to(cuda_device, non_blocking=True)
	seq_batch, seq_o = seq_batch.long(), seq_o.long()
	output = model(seq_batch, seq_o.narrow(1, 0, lo))
	_, trans = torch.max(output, -1)
	return trans

td = h5File(cnfg.test_data, "r")

ntest = td["ndata"][()].item()
nwordi = td["nword"][()].tolist()[0]
vcbt, nwordt = ldvocab(sys.argv[2])
vcbt = reverse_dict(vcbt)

mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize)

mymodel.load_state_dict(h5load(sys.argv[1]))

mymodel.eval()

use_cuda = cnfg.use_cuda
if use_cuda and torch.cuda.is_available():
	use_cuda = True

if cuda_device:
	mymodel.to(cuda_device, non_blocking=True)

beam_size = cnfg.beam_size

ens = "\n".encode("utf-8")

#td_src_grp = td["src"]
with open(sys.argv[3], "wb") as f:
	for i in tqdm(range(ntest), mininterval=tqdm_mininterval):
		#seq_batch = torch.from_numpy(td_src_grp[str(i)][()])
		#if cuda_device:
			#seq_batch = seq_batch.to(cuda_device, non_blocking=True)
#seq_batch = #seq_batch.long()
		#output = mymodel.decode(seq_batch, beam_size).tolist()
		output = eva(td, i, mymodel).tolist()
		for tran in output:
			tmp = []
			for tmpu in tran:
				if (tmpu == 2) or (tmpu == 0):
					break
				else:
					tmp.append(vcbt[tmpu])
			f.write(" ".join(tmp).encode("utf-8"))
			f.write(ens)

td.close()
