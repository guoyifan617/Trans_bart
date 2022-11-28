#encoding: utf-8

from cnfg.base import *

arch_steps = 10
model_steps = 90
arch_early_stop = 64
arch_cost_weight = 0.2 * (arch_steps + model_steps) / arch_steps

tau_step = None

warm_model_steps = int(max(1, warm_step / (arch_steps + model_steps) * model_steps // 2))
