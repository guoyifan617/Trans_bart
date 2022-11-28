#encoding: utf-8

from math import sqrt

from modules.base import Linear
from modules.elinear import Linear as ELinear
from utils.base import add_module, wrap_float2odd

hid_func_min = lambda isize, osize: min(isize, osize)
hid_func_max = lambda isize, osize: max(isize, osize)

@wrap_float2odd
def hid_func_ratio(isize, osize, ratio=1.5):

	return max(isize, osize) * ratio

@wrap_float2odd
def hid_func_mean(isize, osize):

	return float(isize + osize) / 2.0

@wrap_float2odd
def hid_func_sqmean(isize, osize):

	return sqrt(float(isize * osize))

def extend_linear(netin, dim_limit=5120, hid_func=None, nbias=1):

	for _name, _module in netin.named_modules():
		if isinstance(_module, Linear):
			osize, isize = _module.weight.size()
			if max(osize, isize) <= dim_limit:
				use_bias = not (_module.bias is None)
				add_module(netin, _name, ELinear(isize, osize, bias=use_bias, hidden_features=None if hid_func is None else hid_func(isize, osize), nbias=nbias))

	return netin

def std_linear(netin):

	for _name, _module in netin.named_modules():
		if isinstance(_module, ELinear):
			add_module(netin, _name, _module.to_std())

	return netin
