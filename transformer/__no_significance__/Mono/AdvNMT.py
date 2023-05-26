#encoding: utf-8

from modules.mono import FFDiscriminator, GradientBalanceLayer
from transformer.Encoder import Encoder
from transformer.Mono.Decoder import Decoder
from transformer.NMT import NMT as NMTBase
from utils.fmt.parser import parse_double_value_tuple
from utils.relpos.base import share_rel_pos_cache
from utils.torch.comp import torch_no_grad
from utils.train.base import freeze_module, unfreeze_module

from cnfg.ihyp import *
from cnfg.vocab.mono import pad_id

class NMT(NMTBase):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, adv_weight=1.0, clip_value=0.01, num_layer_adv=None, **kwargs):

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		super(NMT, self).__init__(isize, snwd, tnwd, (enc_layer, dec_layer,), fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=forbidden_index)

		self.enc = Encoder(isize, snwd, enc_layer, fhsize, dropout, attn_drop, act_drop, num_head, xseql, ahsize, norm_output)

		emb_w = self.enc.wemb.weight if global_emb else None

		self.dec = Decoder(isize, tnwd, dec_layer, fhsize, dropout, attn_drop, act_drop, emb_w, num_head, xseql, ahsize, norm_output, bindDecoderEmb, forbidden_index)

		self.disc = FFDiscriminator(_isize, num_layer if num_layer_adv is None else num_layer_adv, _dropout)
		self.gbl = GradientBalanceLayer(_adv_weight)
		self.clip_value = clip_value
		self.training_advers = True

		if rel_pos_enabled:
			share_rel_pos_cache(self)

	def forward(self, inpute, inputo, mask=None, lang_id=0, psind=None, **kwargs):

		_mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask

		ence = self.enc(inpute, _mask)

		if self.training:
			_disc_mask = _mask.squeeze(1).unsqueeze(-1)
			if self.training_advers:
				scores = self.disc(ence)
			else:
				enced, ence = self.gbl(ence, _disc_mask)
				scores = self.disc(enced)
			scores.masked_fill_(_disc_mask, 0.0)
			scores = scores.sum()
			if lang_id == 0:
				scores = -scores
			if self.training_advers:
				return None, scores
			else:
				return self.dec(ence, inputo, _mask, lang_id=lang_id, psind=psind), scores
		else:
			return self.dec(ence, inputo, _mask, lang_id=lang_id, psind=psind)

	def decode(self, inpute, beam_size=1, max_len=None, length_penalty=0.0, lang_id=0, **kwargs):

		mask = inpute.eq(pad_id).unsqueeze(1)

		_max_len = (inpute.size(1) + max(64, inpute.size(1) // 4)) if max_len is None else max_len

		return self.dec.decode(self.enc(inpute, mask), mask, beam_size, _max_len, length_penalty, lang_id)

	def fix_init(self):

		self.fix_update()

	def fix_update(self):

		with torch_no_grad():
			for para in self.disc.get_clip_para():
				para.clamp_(-self.clip_value, self.clip_value)

	def train_advers(self, mode=True):

		if mode:
			freeze_module(self.enc)
			freeze_module(self.dec)
			unfreeze_module(self.disc)
			self.disc.train()
		else:
			unfreeze_module(self.enc)
			unfreeze_module(self.dec)
			freeze_module(self.disc)
			self.disc.eval()
		self.training_advers = mode

	def mt_parameters(self):

		return nn.ModuleList([self.enc, self.dec]).parameters()

	def adv_parameters(self):

		return self.disc.parameters()

	def load_base(self, base_mass):

		self.enc = base_mass.enc
		self.dec = base_mass.dec
