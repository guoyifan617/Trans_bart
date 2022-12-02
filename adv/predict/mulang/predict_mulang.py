#encoding: utf-8

import sys
import torch

from parallel.parallelMT import DataParallelMT
from transformer.EnsembleNMT import NMT as Ensemble
from transformer.MuLang.NMT import NMT
from utils.base import set_random_seed
from utils.fmt.base import ldvocab, reverse_dict
from utils.fmt.base4torch import parse_cuda_decode
from utils.h5serial import h5File
from utils.io import load_model_cpu
from utils.torch.comp import torch_autocast, torch_no_grad
from utils.tqdm import tqdm

import cnfg.mulang as cnfg
from cnfg.ihyp import *
from cnfg.vocab.base import eos_id

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

td = h5File(cnfg.test_data, "r")

ntest = td["ndata"][()].item()
nwordi, ntask = td["nword"][()].tolist()
vcbt, nwordt = ldvocab(sys.argv[2])
vcbt = reverse_dict(vcbt)

if len(sys.argv) == 4:
	mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes, ntask=ntask, ngroup=cnfg.ngroup)

	mymodel = load_model_cpu(sys.argv[3], mymodel)
	mymodel.apply(load_fixing)

else:
	models = []
	for modelf in sys.argv[3:]:
		tmp = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes, ntask=ntask, ngroup=cnfg.ngroup)

		tmp = load_model_cpu(modelf, tmp)
		tmp.apply(load_fixing)

		models.append(tmp)
	mymodel = Ensemble(models)

mymodel.eval()

use_cuda, cuda_device, cuda_devices, multi_gpu = parse_cuda_decode(cnfg.use_cuda, cnfg.gpuid, cnfg.multi_gpu_decoding)
use_amp = cnfg.use_amp and use_cuda

set_random_seed(cnfg.seed, use_cuda)

if cuda_device:
	mymodel.to(cuda_device, non_blocking=True)
	if multi_gpu:
		mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=False)

beam_size = cnfg.beam_size
length_penalty = cnfg.length_penalty

ens = "\n".encode("utf-8")

src_grp, task_grp = td["src"], td["task"]
with open(sys.argv[1], "wb") as f, torch_no_grad():
	for i in tqdm(range(ntest), mininterval=tqdm_mininterval):
		bid = str(i)
		seq_batch = torch.from_numpy(src_grp[bid][()])
		seq_t = torch.from_numpy(task_grp[bid][()])
		if cuda_device:
			seq_batch = seq_batch.to(cuda_device, non_blocking=True)
			seq_t = seq_t.to(cuda_device, non_blocking=True)
		seq_batch, seq_t = seq_batch.long(), seq_t.long()
		with torch_autocast(enabled=use_amp):
			output = mymodel.decode(seq_batch, seq_t, beam_size, None, length_penalty)
		if multi_gpu:
			tmp = []
			for ou in output:
				tmp.extend(ou.tolist())
			output = tmp
		else:
			output = output.tolist()
		for tran in output:
			tmp = []
			for tmpu in tran:
				if tmpu == eos_id:
					break
				else:
					tmp.append(vcbt[tmpu])
			f.write(" ".join(tmp).encode("utf-8"))
			f.write(ens)

td.close()