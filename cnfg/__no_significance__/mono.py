#encoding: utf-8

from cnfg.base import *

train_data_src = "cache/"+data_id+"/train_src.mass.h5"
train_data_tgt = "cache/"+data_id+"/train_tgt.mass.h5"
dev_data = "cache/"+data_id+"/dev.mass.h5"

forbidden_indexes = (0, 1, 3,)
tokens_optm = 3000

mask_ratio = 0.8
random_ratio = 0.1
len_ratio = 0.5

sample_target = False
sample_ratio = 0.801559

mt_steps = training_steps
adv_steps = mt_steps * 8
adv_add_init_steps_ext = 512

_expand_ratio = (mt_steps + adv_steps) / mt_steps
#save_every = None if save_every is None else max(1, int(save_every * _expand_ratio))
#training_steps = None if training_steps is None else max(1, int(training_steps * _expand_ratio))
#batch_report = None if batch_report is None else max(1, int(batch_report * _expand_ratio))

start_with_adv = False
adv_weight = 1.0
adv_lr = 5e-5
