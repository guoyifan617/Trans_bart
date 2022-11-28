#encoding: utf-8

from torch.nn import ModuleList

from modules.spreader.manual.rnn import Spreader

def share_spreader_cache(netin):

	for net in netin.modules():
		if isinstance(net, ModuleList):
			decay_d = {}
			decay_beta_d = {}
			rel_pos_d = {}
			decay_emb_d = {}
			decay_head_d = {}
			for layer in net.modules():
				if isinstance(layer, Spreader):
					if not layer.decay.requires_grad:
						_key = tuple(layer.decay.tolist())
						if _key in decay_d:
							layer.decay = decay_d[_key]
						else:
							decay_d[_key] = layer.decay
						if hasattr(layer, "decay_beta"):
							if _key in decay_beta_d:
								layer.decay_beta = decay_beta_d[_key]
							else:
								decay_beta_d[_key] = layer.decay_beta
					if hasattr(layer, "rel_pos"):
						_key = tuple(layer.rel_pos[0].tolist())
						if _key in rel_pos_d:
							layer.ref_rel_posm = rel_pos_d[_key]
							layer.rel_pos = rel_pos_d[_key].rel_pos
						else:
							rel_pos_d[_key] = layer
						if hasattr(layer, "decay_emb"):
							if _key in decay_emb_d:
								layer.ref_attn_matm = decay_emb_d[_key]
								layer.decay_emb = decay_emb_d[_key].decay_emb
							else:
								decay_emb_d[_key] = layer
						if hasattr(layer, "decay_head"):
							_key = tuple(layer.decay_head.size())
							if _key in decay_head_d:
								layer.decay_head = decay_head_d[_key]
							else:
								decay_head_d[_key] = layer.decay_head

	return netin
