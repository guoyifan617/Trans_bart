#encoding: utf-8

import torch

def clear_pad_mask(batch_in, mask, dim=-1):

	npad = mask.int().sum(dim).min().item()
	if npad > 0:
		seql = batch_in.size(dim)
		return batch_in.narrow(dim, 0, seql - npad), mask.narrow(dim, 0, seql - npad)
	else:
		return batch_in, mask

#ent: (bsize, nsent, seql, isize)
#mask: (bsize, nsent, seql)
def merge_sents(ent, mask):

	enrs, mrs = zip(*[clear_pad_mask(entu, masku, dim=1) for entu, masku in zip(ent.unbind(1), mask.unbind(1))])

	return torch.cat(enrs, dim=1), torch.cat(mrs, dim=1)
