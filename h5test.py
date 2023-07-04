import h5py
import tqdm
import torch
import sys
from utils.h5serial import h5File
tqdm_mininterval = 1.0
h5files = sys.argv[1]
id = sys.argv[2]
td = h5File(h5files, "r")
ntrain = td["ndata"][()].item()
i_d = str(id)
src_grp, tgt_grp = td["src"], td["tgt"]
seq_batch = torch.from_numpy(src_grp[i_d][()])
seq_o = torch.from_numpy(tgt_grp[i_d][()])
seq_batch = seq_batch.tolist()
seq_o = seq_o.tolist()
print("src_batch is:",'\n',seq_batch)
print("-" * 100)
print("tgt_batch is:",'\n',seq_o)