#encoding: utf-8

from cnfg.base import *

src_tgt_m = "expm/w19ape/mt/src_tgt/avg.h5"
tgt_src_m = "expm/w19ape/mt/tgt_src/avg.h5"

from srcfbind import fbl as srcfbl

tsm_forbidden_indexes = [0, 1] + srcfbl

nlayer_mt = nlayer

eval_decode = True
beam_size = 1
length_penalty = 0.0
