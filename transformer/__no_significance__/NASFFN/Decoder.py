#encoding: utf-8

from math import sqrt
from torch import nn
from transfromer.Decoder import Decoder as DecoderBase

from modules.base import MultiHeadAttn
from modules.nas.ffn import PositionwiseFF, ResCrossAttn, ResSelfAttn, ResTaughtCrossAttn, ResTaughtSelfAttn
from transformer.Decoder import DecoderLayer as DecoderLayerBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class DecoderLayer(DecoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, norm_residual=norm_residual_default, k_rel_pos=use_k_relative_position_decoder, max_bucket_distance=relative_position_max_bucket_distance_decoder, search_ffn=False, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(DecoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, norm_residual=norm_residual, k_rel_pos=k_rel_pos, max_bucket_distance=max_bucket_distance, **kwargs)

		if search_ffn:
			self.ff = PositionwiseFF(isize, hsize=_fhsize, dropout=dropout, act_drop=act_drop, norm_residual=norm_residual)
		self.self_attn = ResTaughtSelfAttn(isize, _ahsize, num_head=num_head, dropout=attn_drop, norm_residual=norm_residual, k_rel_pos=k_rel_pos, uni_direction_reduction=True, max_bucket_distance=max_bucket_distance)
		self.cross_attn = ResTaughtCrossAttn(isize, _ahsize, num_head=num_head, dropout=attn_drop, norm_residual=norm_residual)

	def forward(self, inpute, inputo, sattn, cattn, query_unit=None, **kwargs):

		if query_unit is None:
			context = self.self_attn(inputo, sattn)
		else:
			context, states_return = self.self_attn(query_unit, sattn, states=inputo)

		context = self.cross_attn(context, inpute, cattn)

		context = self.ff(context)

		if query_unit is None:
			return context
		else:
			return context, states_return

	def load_base(self, base_layer):

		self.self_attn.net.load_base(base_layer.self_attn.net)
		self.cross_attn.net.load_base(base_layer.cross_attn.net)

		if isinstance(self.ff, PositionwiseFF):
			self.ff.load_base(base_layer.ff)
		else:
			self.ff = base_layer.ff
		self.layer_normer1, self.layer_normer2, self.drop, self.norm_residual = base_layer.layer_normer1, base_layer.layer_normer2, base_layer.drop, base_layer.norm_residual

class StdDecoderLayer(DecoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, norm_residual=norm_residual_default, k_rel_pos=use_k_relative_position_decoder, max_bucket_distance=relative_position_max_bucket_distance_decoder, **kwargs):

		_ahsize = parse_none(ahsize, isize)

		super(StdDecoderLayer, self).__init__(isize, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, norm_residual=norm_residual, k_rel_pos=k_rel_pos, max_bucket_distance=max_bucket_distance)

		self.self_attn = ResSelfAttn(isize, _ahsize, num_head=num_head, dropout=attn_drop, norm_residual=norm_residual, k_rel_pos=k_rel_pos, uni_direction_reduction=True, max_bucket_distance=max_bucket_distance)
		self.cross_attn = ResCrossAttn(isize, _ahsize, num_head=num_head, dropout=attn_drop, norm_residual=norm_residual)

	def forward(self, inpute, inputo, src_pad_mask=None, tgt_pad_mask=None, query_unit=None, **kwargs):

		if query_unit is None:
			context, sattn = self.self_attn(inputo, mask=tgt_pad_mask)
		else:
			context, states_return, sattn = self.self_attn(query_unit, states=inputo)

		context, cattn = self.cross_attn(context, inpute, mask=src_pad_mask)

		context = self.ff(context)

		if query_unit is None:
			return context, sattn, cattn
		else:
			return context, sattn, cattn, states_return

	def load_base(self, base_layer):

		self.self_attn.net.load_base(base_layer.net.self_attn)
		self.cross_attn.net.load_base(base_layer.net.cross_attn)

		self.ff, self.self_attn.normer, self.cross_attn.normer, self.self_attn.norm_residual, self.cross_attn.norm_residual = base_layer.ff, base_layer.self_attn.normer, base_layer.cross_attn.normer, base_layer.self_attn.norm_residual, base_layer.cross_attn.norm_residual

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, share_layer=False, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, share_layer=False, **kwargs)

		if share_layer:
			_shared_layer = DecoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
			_shared_layer = StdDecoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize)
			self.stdnets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([DecoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize) for i in range(num_layer)])
			self.stdnets = nn.ModuleList([StdDecoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize) for i in range(num_layer)])

	def forward(self, inpute, stdinpute, inputo, src_pad_mask=None, **kwargs):

		nquery = inputo.size(-1)

		out = self.wemb(inputo)

		if self.pemb is not None:
			out = self.pemb(inputo, expand=False).add(out, alpha=sqrt(out.size(-1)))
		if self.drop is not None:
			out = self.drop(out)

		stdout = out
		_mask = self._get_subsequent_mask(nquery)

		for snet, net in zip(self.stdnets, self.nets):
			stdout, sattn, cattn = snet(stdinpute, stdout, src_pad_mask, _mask)
			out = net(inpute, out, sattn, cattn)

		if self.out_normer is not None:
			out = self.out_normer(out)

		out = self.lsm(self.classifier(out))

		return out

	def load_base(self, base_decoder):

		self.drop = base_decoder.drop

		self.wemb = base_decoder.wemb

		self.pemb = base_decoder.pemb

		for net, snet, bnet in zip(self.nets, self.stdnets, base_decoder.nets):
			net.load_base(bnet)
			snet.load_base(bnet)

		self.classifier = base_decoder.classifier

		self.lsm = base_decoder.lsm

		self.out_normer = None if self.out_normer is None else base_decoder.out_normer

	def repeat_cross_attn_buffer(self, beam_size):

		for _m in self.modules():
			if isinstance(_m, (CrossAttn, MultiHeadAttn, TaughtCrossAttn,)):
				_m.repeat_buffer(beam_size)
