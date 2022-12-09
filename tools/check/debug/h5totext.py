#encoding: utf-8

# python h5totext.py $data.h5 $vocab.vcb $group_name $rsf

import sys

from utils.fmt.vocab.base import reverse_dict
from utils.fmt.vocab.token import ldvocab
from utils.h5serial import h5File

from cnfg.vocab.base import eos_id

with h5File(sys.argv[1], "r") as td:
	ntest = td["ndata"][()].item()
	nwordi = td["nword"][()].tolist()[0]
	vcbt, nwordt = ldvocab(sys.argv[2])
	vcbt = reverse_dict(vcbt)

	ens = "\n".encode("utf-8")

	src_grp = td[sys.argv[3]]
	with open(sys.argv[4], "wb") as f:
		for i in range(ntest):
			seq_batch = src_grp[str(i)][()].tolist()
			for tran in output:
				tmp = []
				for tmpu in tran[1:]:
					if tmpu == eos_id:
						break
					else:
						tmp.append(vcbt[tmpu])
				f.write(" ".join(tmp).encode("utf-8"))
				f.write(ens)
