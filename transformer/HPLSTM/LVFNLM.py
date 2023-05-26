#encoding: utf-8

from modules.adaptive import AdaptiveEmbedding, ProjectedAdaptiveLogSoftmax
from transformer.HPLSTM.FNLM import Decoder as DecoderBase
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=True, forbidden_index=None, share_layer=False, cutoffs=[20000, 40000, 200000], ignore_index=0, tie_projs=True, disable_pemb=disable_std_pemb_decoder, **kwargs):

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, share_layer=share_layer, disable_pemb=True, **kwargs)

		self.wemb = AdaptiveEmbedding(nwd, isize, cutoffs, padding_idx=pad_id)
		self.classifier = ProjectedAdaptiveLogSoftmax(nwd, isize, cutoffs, ignore_index=ignore_index, reduction="sum")

		if bindemb:
			for i in range(len(self.classifier.out_layers)):
				self.classifier.out_layers[i].weight = self.wemb.emb_layers[i].weight
		if tie_projs:
			for i in range(len(self.classifier.out_projs)):
				if self.classifier.div_val == 1 and self.classifier.d_model != self.classifier.d_embed:
					self.classifier.out_projs[i] = self.wemb.emb_projs[0]
				elif self.classifier.div_val != 1:
					self.classifier.out_projs[i] = self.wemb.emb_projs[i]

		self.lsm = None

	def forward(self, inputo, gold, states=None, **kwargs):

		out = self.wemb(inputo)

		if self.drop is not None:
			out = self.drop(out)

		_states = {} if states is None else states
		states_return = {}
		for i, net in enumerate(self.nets):
			_state = _states.get(i, "init")
			out, _state = net(_state, query_unit=out)
			states_return[i] = _state

		if self.out_normer is not None:
			out = self.out_normer(out)

		out = self.classifier(out.view(-1, out.size(-1)), gold.view(-1))

		return out, states_return

	def fix_load(self):

		if self.fbl is not None:
			with torch_no_grad():
				fblt = torch.as_tensor(self.fbl, dtype=torch.long, device=self.classifier.out_layers[0].bias.device)
				if self.classifier.div_val == 1:
					self.classifier.out_layers[0].bias.index_fill_(0, fblt, -inf_default)
				else:

					for i in range(len(self.classifier.cutoffs)):
						l_idx, r_idx = self.classifier.cutoff_ends[i], self.classifier.cutoff_ends[i + 1]
						mask = (fblt >= l_idx) & (fblt < r_idx)
						sel_fblt = fblt.masked_select(mask)
						if sel_fblt.numel() > 0:
							self.classifier.out_layers[i].bias.index_fill_(0, sel_fblt - l_idx, -inf_default)
