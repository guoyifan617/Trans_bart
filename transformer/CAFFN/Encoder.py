#encoding: utf-8

from torch import nn

#from modules.caffn import BiCFFN as PositionwiseFF
from modules.hplstm.hfn import BiHPLSTM
from transformer.Encoder import Encoder as EncoderBase, EncoderLayer as EncoderLayerBase
from utils.fmt.parser import parse_none
from utils.torch.comp import flip_mask

from cnfg.ihyp import *

class EncoderLayer(EncoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, norm_residual=norm_residual_default, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(EncoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, norm_residual=norm_residual, **kwargs)

		self.ff = BiHPLSTM(isize, num_head=num_head, osize=isize, fhsize=_fhsize, dropout=dropout, act_drop=act_drop)#PositionwiseFF(isize, hsize=_fhsize, dropout=dropout, act_drop=act_drop, norm_residual=norm_residual)

	def forward(self, inputs, mask=None, reversed_mask=None, **kwargs):

		context = self.attn(inputs, mask=mask)

		_context = self.ff(context, reversed_mask=reversed_mask)
		if self.drop is not None:
			_context = self.drop(_context)

		return context + _context

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, share_layer=False, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, share_layer=share_layer, **kwargs)

		if share_layer:
			_shared_layer = EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize) for i in range(num_layer)])

	def forward(self, inputs, mask=None, **kwargs):

		_rmask = None if mask is None else flip_mask(mask.view(*(list(inputs.size())+[1, 1])), 1)
		out = self.wemb(inputs)

		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(out, mask=mask, reversed_mask=_rmask)

		return out if self.out_normer is None else self.out_normer(out)
