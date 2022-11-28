#encoding: utf-8

from utils.torch.comp import torch_no_grad

def regularize_grad(base_para_g, model_para_g, weight=0.2, p=2, ieps=1e-6):

	with torch_no_grad():
		for base_p, model_p in zip(base_para_g, model_para_g):
			if model_p.requires_grad and (model_p.grad is not None):
				_reg_g = model_p.data - base_p.data
				model_p.grad.add_(_reg_g, alpha=weight * (model_p.grad.norm(p=p) / (_reg_g.norm(p=p) + ieps)).item())

	return model_para_g

def backup_para_data(plin):

	with torch_no_grad():
		rs = [pu.data.clone() for pu in plin if pu.requires_grad]

	return rs
