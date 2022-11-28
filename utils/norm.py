#encoding: utf-8

from torch.nn import LayerNorm

from modules.norm import SimpNorm
from utils.base import add_module

def disable_ln_bias(modin):

	for _m in modin.modules():
		if isinstance(_m, LayerNorm):
			_m.register_parameter("bias", None)

	return modin

def simplify_ln(modin):

	for _name, _module in modin.named_modules():
		if isinstance(_module, LayerNorm):
			_tmpm = SimpNorm(_module.normalized_shape if _module.weight is None else tuple(_module.weight.size()), eps=_module.eps, elementwise_affine=_module.elementwise_affine)
			_tmpm.load_base(_module)
			add_module(modin, _name, _tmpm)

	return modin
