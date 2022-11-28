#encoding: utf-8

from math import sqrt
from torch import nn

from modules.mono import PositionalEmb
from transformer.Doc.Para.Base.Encoder import CrossEncoder as CrossEncoderBase
from transformer.MDoc.Encoder import Encoder as PretEncoder

from cnfg.ihyp import *

class CrossEncoder(CrossEncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, nprev_context=1, **kwargs):

		super(CrossEncoder, self).__init__(isize, nwd, num_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, nprev_context=nprev_context)

		self.pemb = PositionalEmb(isize, xseql, 0, 0)
		self.semb = PositionalEmb(isize, xseql, 0, 0)#5000, isize or 5000, be consistent with MDoc.Encoder

	def forward(self, inputs, inputc, mask=None, context_mask=None, start_sent_id=0, **kwargs):

		out = self.wemb(inputs)
		seql, isize = out.size()[2:]
		out = self.pemb(inputs.select(1, 0), expand=False).unsqueeze(1) + self.semb(inputs.select(-1, 0), expand=False, sind=start_sent_id).unsqueeze(2) + out * sqrt(isize)
		out = out.view(-1, seql, isize)

		if self.drop is not None:
			out = self.drop(out)

		_mask = None if mask is None else mask.view(-1, 1, seql)

		for net in self.nets:
			out = net(out, inputc, _mask, context_mask)

		return out if self.out_normer is None else self.out_normer(out)

	def load_base(self, base_encoder):

		self.drop = base_encoder.drop

		self.wemb = base_encoder.wemb

		for snet, bnet in zip(self.nets, base_encoder.nets):
			snet.load_base(bnet)

		self.out_normer = None if self.out_normer is None else base_encoder.out_normer

	def get_loaded_para_models(self):

		rs = [self.wemb]
		if self.out_normer is not None:
			rs.append(self.out_normer)
		for net in self.nets:
			rs.extend([net.attn, net.ff, net.layer_normer])

		return rs

class Encoder(nn.Module):

	def __init__(self, isize, nwd, pnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, num_layer_pret=12, **kwargs):

		super(Encoder, self).__init__()

		_ahsize = isize if ahsize is None else ahsize

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		self.context_enc = PretEncoder(isize, pnwd, num_layer if num_layer_pret is None else num_layer_pret, _fhsize, dropout, attn_drop, num_head, xseql, _ahsize, norm_output)
		self.enc = CrossEncoder(isize, nwd, num_layer, _fhsize, dropout, attn_drop, num_head, xseql, _ahsize, norm_output, 1)

	# inputs: (bsize, nsent, seql)
	# inputc: (bsize, nsent, seql)
	# mask: (bsize, 1, nsent, seql), generated with:
	#	mask = inputs.eq(0).unsqueeze(1)
	def forward(self, inputs, inputc, mask=None, context_mask=None, start_sent_id=0, **kwargs):

		bsize, nsent_doc = inputc.size()[:2]
		nsent, seql = inputs.size()[1:]
		context = [self.context_enc(inputc, mask=context_mask).repeat(1, nsent, 1).view(bsize * nsent, nsent_doc, -1)]

		return self.enc(inputs, context, mask, None, start_sent_id=start_sent_id), context

	def load_base(self, base_encoder, base_encoder_pret=None):

		self.enc.load_base(base_encoder)
		if base_encoder_pret is not None:
			self.context_enc = base_encoder_pret

	def get_loaded_para_models(self):

		rs = self.enc.get_loaded_para_models()
		rs.append(self.context_enc)

		return rs

	def update_vocab(self, indices):

		self.context_enc.update_vocab(indices)
		self.enc.update_vocab(indices)
