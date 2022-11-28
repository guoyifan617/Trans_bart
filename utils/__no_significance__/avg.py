#encoding: utf-8

import torch

from utils.torch.comp import torch_no_grad

def avg_paras(rspl, plist):

	with torch_no_grad():
		for prs, pl in zip(rspl, zip(*plist)):
			prs.data = torch.stack(list(pl) + [prs.data], dim=-1).mean(-1)

	return rspl
