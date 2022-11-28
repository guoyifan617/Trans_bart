#encoding: utf-8

import torch
from math import sqrt
from torch.optim.optimizer import Optimizer

from utils.angle import prep_cos
from utils.torch.comp import torch_no_grad

class Adam(Optimizer):

	def __init__(self, params, lr=1e-3, betas=(0.9, 0.999,), eps=1e-8, weight_decay=0, amsgrad=False, thres=1e-3, **kwargs):
		defaults = dict(lr=lr, betas=(1.0 - betas[0], betas[-1],), eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, thres=float(thres))
		super(Adam, self).__init__(params, defaults)
		self.inited = False

	@torch_no_grad()
	def step(self, closure=None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		for group in self.param_groups:
			lr, thres = group["lr"], group["thres"]
			p_norm = torch.stack([p.data.pow(2).sum() for p in group["params"] if p.grad is not None], 0).sum().sqrt().item()
			if self.inited:
				on, o, n = zip(*[prep_cos(self.state[p]["exp_avg"], p.grad) for p in group["params"] if p.grad is not None])
				exp_avg_norm, grad_norm = torch.stack(o, 0).sum().sqrt().item(), torch.stack(n, 0).sum().sqrt().item()
				cos_sim = (torch.stack(on, 0).sum().item() / (exp_avg_norm * grad_norm))
				if cos_sim <= 0.0:
					copy_grad = True
					_r = grad_norm * lr / p_norm / thres
					if _r > 1.0:
						lr /= _r
				else:
					copy_grad = False
					sin_sim = sqrt(1.0 - cos_sim * cos_sim)
					r = sin_sim * exp_avg_norm / grad_norm
					beta, _ = group["betas"]
					if r <= beta:
						r = None
						_r = (grad_norm + exp_avg_norm) * lr / p_norm / thres
						if _r > 1.0:
							lr /= _r
					else:
						r = beta / r
						_r = (grad_norm + exp_avg_norm * r) * lr / p_norm / thres
						if _r > 1.0:
							lr /= _r
			for p in group["params"]:
				if p.grad is not None:
					grad = p.grad
					amsgrad = group["amsgrad"]

					state = self.state[p]

					if len(state) == 0:
						state["step"] = 0
						state["exp_avg"] = p.grad.clone()
						state["exp_avg_sq"] = p.new_zeros(p.size())
						if amsgrad:
							state["max_exp_avg_sq"] = p.new_zeros(p.size())
						nonzero_state = False
						self.inited = True
					else:
						nonzero_state = False

					exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
					if amsgrad:
						max_exp_avg_sq = state["max_exp_avg_sq"]
					beta1, beta2 = group["betas"]

					state["step"] += 1
					bias_correction2 = 1 - beta2 ** state["step"]

					if group["weight_decay"] != 0:
						grad = grad.add(p, alpha=group["weight_decay"])
					if nonzero_state:
						if copy_grad:
							exp_avg.copy_(grad)
						else:
							if r is None:
								exp_avg.add_(grad)
							else:
								exp_avg.mul_(r).add_(grad)
					exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
					if amsgrad:
						torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
						denom = (max_exp_avg_sq.sqrt() / sqrt(bias_correction2)).add_(group["eps"])
					else:
						denom = (exp_avg_sq.sqrt() / sqrt(bias_correction2)).add_(group["eps"])

					p.addcdiv_(exp_avg, denom, value=-lr)

		return loss
