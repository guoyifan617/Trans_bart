#encoding: utf-8

import sys
import torch

from parallel.parallelMT import DataParallelMT
from transformer.EnsembleNMT import NMT as Ensemble
from transformer.PretDoc.NMT import NMT
from utils.base import set_random_seed
from utils.doc.pret.base4torch import patch_pret_model_ffn
from utils.fmt.base4torch import parse_cuda_decode
from utils.fmt.vocab.base import reverse_dict
from utils.fmt.vocab.token import ldvocab
from utils.h5serial import h5File
from utils.io import load_model_cpu
from utils.torch.comp import torch_autocast, torch_inference_mode
from utils.tqdm import tqdm

import cnfg.docpret as cnfg
from cnfg.ihyp import *
from cnfg.vocab.base import eos_id

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

td = h5File(cnfg.test_data, "r")

tl = [(str(nsent), str(_curd),) for nsent, ndata in zip(td["nsent"][()].tolist(), td["ndata"][()].tolist()) for _curd in range(ndata)]
nword = td["nword"][()].tolist()
nwordi, nwordp = nword[0], nword[1]
vcbt, nwordt = ldvocab(sys.argv[2])
vcbt = reverse_dict(vcbt)

if len(sys.argv) == 4:
	mymodel = NMT(cnfg.isize, nwordi, nwordp, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes, cnfg.num_layer_pret)
	mymodel.enc.context_enc = patch_pret_model_ffn(mymodel.enc.context_enc)
	mymodel = load_model_cpu(sys.argv[3], mymodel)
	mymodel.apply(load_fixing)

else:
	models = []
	for modelf in sys.argv[3:]:
		tmp = NMT(cnfg.isize, nwordi, nwordp, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes, cnfg.num_layer_pret)
		tmp.enc.context_enc = patch_pret_model_ffn(tmp.enc.context_enc)
		tmp = load_model_cpu(modelf, tmp)
		tmp.apply(load_fixing)

		models.append(tmp)
	mymodel = Ensemble(models)

mymodel.eval()

use_cuda, cuda_device, cuda_devices, multi_gpu = parse_cuda_decode(cnfg.use_cuda, cnfg.gpuid, cnfg.multi_gpu_decoding)
use_amp = cnfg.use_amp and use_cuda

# Important to make cudnn methods deterministic
set_random_seed(cnfg.seed, use_cuda)

if cuda_device:
	mymodel.to(cuda_device, non_blocking=True)
	if multi_gpu:
		mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=False)

beam_size = cnfg.beam_size
length_penalty = cnfg.length_penalty

ens = "\n".encode("utf-8")

src_grp, pret_grp = td["src"], td["pret"]
with open(sys.argv[1], "wb") as f, torch_inference_mode():
	for nsent, i_d in tqdm(tl, mininterval=tqdm_mininterval):
		seq_batch = torch.from_numpy(src_grp[nsent][i_d][()])
		seq_pret = torch.from_numpy(pret_grp[nsent][i_d][()])
		if cuda_device:
			seq_batch = seq_batch.to(cuda_device, non_blocking=True)
			seq_pret = seq_pret.to(cuda_device, non_blocking=True)
		seq_batch, seq_pret = seq_batch.long(), seq_pret.long()
		bsize, nsent, seql = seq_batch.size()
		with torch_autocast(enabled=use_amp):
			output = mymodel.decode(seq_batch, seq_pret, beam_size, None, length_penalty).view(bsize, nsent, -1)
		if multi_gpu:
			tmp = []
			for ou in output:
				tmp.extend(ou.tolist())
			output = tmp
		else:
			output = output.tolist()
		for doc in output:
			for tran in doc:
				tmp = []
				for tmpu in tran:
					if tmpu == eos_id:
						break
					else:
						tmp.append(vcbt[tmpu])
				f.write(" ".join(tmp).encode("utf-8"))
				f.write(ens)
			f.write(ens)

td.close()
