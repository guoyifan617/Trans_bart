#encoding: utf-8

from modules.advdrop import AdvDropout
from modules.base import CrossAttn, MultiHeadAttn, SelfAttn

def patch_drop_ffn(ffnin):

	_net = ffnin.net
	if len(_net) == 5:
		_net[2] = AdvDropout(_net[2].p, _net[0].weight.size(0), dim=-1)
		_net[-1] = AdvDropout(_net[-1].p, _net[-2].weight.size(0), dim=-1)
	ffnin.net = _net

	return ffnin

def patch_drop_attn(netin):

	for net in netin.modules():
		if isinstance(net, (MultiHeadAttn, SelfAttn, CrossAttn,)):
			if net.drop is not None:
				net.drop = AdvDropout(net.drop.p, net.num_head, dim=1)

	return netin
