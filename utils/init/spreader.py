#encoding: utf-8

import torch

from utils.math import exp_grow as grow_func_py
from utils.torch.ext import exp_grow as grow_func_th

def build_spread_vector_py(start, end, k, f=0.5):

	return [f ** (1.0 / _tmp) for _tmp in grow_func_py(start, end, k)]

def build_spread_vector_th(start, end, k, f=0.5):

	return torch.pow(f, 1.0 / grow_func_th(start, end, k))

build_spread_vector = build_spread_vector_th
