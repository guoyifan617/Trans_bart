#encoding: utf-8

import torch
from math import sqrt
from torch.optim.optimizer import Optimizer

from utils.angle import prep_cos
from utils.torch.comp import torch_no_grad

class Momentum(Optimizer):

	def __init__(self, params, lr=0.1, weight_decay=0, beta=0.1, thres=1e-3, **kwargs):

		defaults = dict(lr=lr, weight_decay=weight_decay, beta=float(beta), thres=float(thres))
		super(Momentum, self).__init__(params, defaults)
		self.inited = False

	@torch_no_grad()
	def step(self, closure=None):

		if closure is None:
			loss = None
		else:
			with torch.enable_grad():
				loss = closure()

		for group in self.param_groups:

			weight_decay, lr, thres = group["weight_decay"], group["lr"], group["thres"]
			weight_decay_lr = weight_decay * lr
			p_norm = torch.stack([p.data.pow(2).sum() for p in group["params"] if p.grad is not None], 0).sum().sqrt().item()
			if self.inited:
				on, o, n = zip(*[prep_cos(self.state[p]["exp_avg"], p.grad) for p in group["params"] if p.grad is not None])
				exp_avg_norm, grad_norm = torch.stack(o, 0).sum().sqrt().item(), torch.stack(n, 0).sum().sqrt().item()
				cos_sim = (torch.stack(on, 0).sum().item() / (exp_avg_norm * grad_norm))
				if cos_sim <= 0.0:
					_r = grad_norm * lr / p_norm / thres
					if _r > 1.0:
						lr /= _r
					for p in group["params"]:
						if p.grad is not None:
							self.state[p]["exp_avg"].copy_(p.grad)
							p.data.add_(p.grad, alpha=-lr)
							if weight_decay > 0.0:
								p.data.add_(-weight_decay_lr, p.data)
				else:
					sin_sim = sqrt(1.0 - cos_sim * cos_sim)
					r = sin_sim * exp_avg_norm / grad_norm
					beta = group["beta"]
					if r <= beta:
						_r = (grad_norm + exp_avg_norm) * lr / p_norm / thres
						if _r > 1.0:
							lr /= _r
						for p in group["params"]:
							if p.grad is not None:
								exp_avg = self.state[p]["exp_avg"]
								exp_avg.add_(p.grad)
								p.data.add_(exp_avg, alpha=-lr)
								if weight_decay > 0.0:
									p.data.add_(-weight_decay_lr, p.data)
					else:
						r = beta / r
						_r = (grad_norm + exp_avg_norm * r) * lr / p_norm / thres
						if _r > 1.0:
							lr /= _r
						for p in group["params"]:
							if p.grad is not None:
								exp_avg = self.state[p]["exp_avg"]
								exp_avg.mul_(r).add_(p.grad)
								p.data.add_(exp_avg, alpha=-lr)
								if weight_decay > 0.0:
									p.data.add_(-weight_decay_lr, p.data)
			else:
				_r = grad_norm * lr / p_norm / thres
				if _r > 1.0:
					lr /= _r
				for p in group["params"]:
					if p.grad is not None:
						self.state[p]["exp_avg"] = p.grad.clone()
						p.data.add_(p.grad, alpha=-lr)
						if weight_decay > 0.0:
							p.data.add_(-weight_decay_lr, p.data)
		self.inited = True

		return loss
