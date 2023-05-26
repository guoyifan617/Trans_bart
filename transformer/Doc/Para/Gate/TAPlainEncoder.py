#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.base import Dropout
from transformer.Doc.Para.Gate.PlainEncoder import Encoder as EncoderBase
from transformer.Doc.Para.Gate.TARSGEncoder import LOEncoder
from utils.doc.paragate.base4torch import clear_pad_mask
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, nprev_context=2, num_layer_cross=1, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, nprev_context=nprev_context, num_layer_cross=num_layer_cross, **kwargs)

		self.enc = LOEncoder(isize, nwd, num_layer, _fhsize, dropout, attn_drop, act_drop, num_head, xseql, _ahsize, norm_output)

		self.nlo = num_layer + 1
		self.tattn_w = nn.Parameter(torch.Tensor(self.nlo).uniform_(- sqrt(1.0 / self.nlo), sqrt(1.0 / self.nlo)))
		self.tattn_drop = Dropout(dropout) if dropout > 0.0 else None

	# inputs: (bsize, nsent, seql), 0, ... , nsent - 1
	# mask: (bsize, 1, nsent, seql), generated with:
	#	mask = inputs.eq(pad_id).unsqueeze(1)
	def forward(self, inputs, mask=None, **kwargs):

		# forward the whole document
		bsize, nsent, seql = inputs.size()
		sbsize = bsize * nsent
		ence_out, ence_ctx = self.enc(inputs.view(sbsize, seql), mask.view(sbsize, 1, seql))
		isize = ence_out.size(-1)
		ence = ence_out.view(bsize, nsent, seql, isize)
		ence_ctx = ence_ctx.view(bsize, nsent, seql, -1).narrow(1, 0, nsent - 1).contiguous().view(-1, self.nlo).mv(self.tattn_w.softmax(dim=0) if self.tattn_drop is None else self.tattn_drop(self.tattn_w).softmax(dim=0)).view(bsize, nsent - 1, seql, -1)
		ence_ctx, mask_ctx = clear_pad_mask(ence_ctx, mask.narrow(2, 0, nsent - 1), dim=2, mask_dim=-1)

		# collect context
		contexts = []
		context_masks = []
		_nsent_context_base = nsent - self.nprev_context
		for i, _sent_pemb in enumerate(self.sent_pemb.unbind(0)):
			_num_context = _nsent_context_base + i
			_context, _mask = clear_pad_mask(ence_ctx.narrow(1, 0, _num_context), mask_ctx.narrow(2, 0, _num_context), dim=2, mask_dim=-1)
			_context = _context + _sent_pemb
			_num_pad = self.nprev_context - i - 1
			if _num_pad > 0:
				_seql = _context.size(2)
				_context = torch.cat((_context.new_zeros((bsize, _num_pad, _seql, isize),), _context,), dim=1)
				_mask = torch.cat((_mask.new_ones((bsize, 1, _num_pad, _seql,),), _mask,), dim=2)
			contexts.append(_context)
			context_masks.append(_mask)
		context = torch.cat(contexts, dim=2)
		context_mask = torch.cat(context_masks, dim=-1)

		# perform source to context attention
		_nsent_out = nsent - 1
		enc_out, enc_mask = clear_pad_mask(ence.narrow(1, 1, _nsent_out), mask.narrow(2, 1, _nsent_out), dim=2, mask_dim=-1)
		_seql = enc_out.size(2)
		sbsize = bsize * _nsent_out
		enc_out, enc_mask, context, context_mask = enc_out.contiguous().view(sbsize, _seql, isize), enc_mask.contiguous().view(sbsize, 1, _seql), context.view(sbsize, -1, isize), context_mask.view(sbsize, 1, -1)

		# original encoder output, context-aware encoding, context output, encoder mask, context mask
		return ence_out, enc_out, context, enc_mask, context_mask

	def load_base(self, base_encoder):

		self.enc.load_base(base_encoder)
