#encoding: utf-8

from cnfg.base import *

kd_layers = (3, 6,)

min_gold_p = 0.1
num_topk = 64
T = 1.0
min_T = T / 64.0
kd_weight = 1.0
kd_step = 0#warm_step + warm_step + 1

mix_kd = True
iter_kd = True
remove_gold = False
enable_proj = True
