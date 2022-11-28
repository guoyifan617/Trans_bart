#encoding: utf-8

from cnfg.base import *

min_gold_p = 0.1
num_topk = 64
T = 1.0
kd_weight = 1.0
mavg_beta = 0.9
warm_cache_steps = 0
warm_mvavg_steps = 0#warm_step + warm_step
kd_step = 0#warm_step + warm_step + 1

num_cache_topk = num_topk + num_topk
