#encoding: utf-8

import torch
from math import sqrt

from cnfg.ihyp import *

def renorm(x, dim=-1):

	return x.div(x.sum(dim, keepdim=True))

def std_norm(x, dim=-1, eps=ieps_ln_default):

	_std, _mean = torch.std_mean(x, dim=dim, unbiased=False, keepdim=True)#.detach()

	return x.sub(_mean).div_(_std.add(eps))

def mean_norm(x, dim=-1, **kwargs):

	return x.sub(x.mean(dim=dim, keepdim=True))

def rms_norm(x, dim=-1, eps=ieps_ln_default):

	return x.div(x.norm(p=2, dim=dim, keepdim=True).div(sqrt(float(x.size(dim)))).add_(eps))

def identity_norm(x, *args, **kwargs):

	return x
