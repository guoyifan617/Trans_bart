#encoding: utf-8

from transfromer.Encoder import Encoder as EncoderBase

from modules.nas.ffn import PositionwiseFF, ResSelfAttn, ResTaughtSelfAttn
from transformer.Encoder import EncoderLayer as EncoderLayerBase
from utils.fmt.parser import parse_none

from cnfg.ihyp import *

class EncoderLayer(EncoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, norm_residual=norm_residual_default, k_rel_pos=use_k_relative_position_encoder, max_bucket_distance=relative_position_max_bucket_distance_encoder, search_ffn=True, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(EncoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, norm_residual=norm_residual, k_rel_pos=k_rel_pos, max_bucket_distance=max_bucket_distance, **kwargs)

		if search_ffn:
			self.ff = PositionwiseFF(isize, hsize=_fhsize, dropout=dropout, act_drop=act_drop, norm_residual=norm_residual)
		self.attn = ResTaughtSelfAttn(isize, _ahsize, num_head=num_head, norm_residual=norm_residual, k_rel_pos=k_rel_pos, max_bucket_distance=max_bucket_distance)

	def forward(self, inputs, attn, **kwargs):

		context = self.attn(inputs, attn)

		context = self.ff(context)

		return context

	def load_base(self, base_layer):

		self.attn.load_base(base_layer.attn)

		if isinstance(self.ff, PositionwiseFF):
			self.ff.load_base(base_layer.ff)
		else:
			self.ff = base_layer.ff
		self.ff, self.layer_normer, self.drop, self.norm_residual = base_layer.ff, base_layer.layer_normer, base_layer.drop, base_layer.norm_residual

class StdEncoderLayer(EncoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, norm_residual=norm_residual_default, k_rel_pos=use_k_relative_position_encoder, max_bucket_distance=relative_position_max_bucket_distance_encoder, **kwargs):

		_ahsize = parse_none(ahsize, isize)

		super(StdEncoderLayer, self).__init__(isize, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, norm_residual=norm_residual, k_rel_pos=k_rel_pos, max_bucket_distance=max_bucket_distance)

		self.attn = ResSelfAttn(isize, _ahsize, num_head=num_head, dropout=attn_drop, norm_residual=norm_residual, k_rel_pos=k_rel_pos, max_bucket_distance=max_bucket_distance)

	def forward(self, inputs, mask=None, **kwargs):

		context, attns = self.attn(inputs, mask=mask)

		context = self.ff(context)

		return context, attns

	def load_base(self, base_layer):

		self.attn.load_base(base_layer.attn)

		self.ff, self.layer_normer, self.drop, self.norm_residual = base_layer.ff, base_layer.layer_normer, base_layer.drop, base_layer.norm_residual

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, **kwargs)

		if share_layer:
			_shared_layer = EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
			_shared_layer = StdEncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize)
			self.stdnets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([EncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize) for i in range(num_layer)])
			self.stdnets = nn.ModuleList([StdEncoderLayer(isize, _fhsize, dropout, attn_drop, act_drop, num_head, _ahsize) for i in range(num_layer)])

	def forward(self, inputs, mask=None, **kwargs):

		out = self.wemb(inputs)
		if self.pemb is not None:
			out = self.pemb(inputs, expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		stdout = out

		for snet, net in zip(self.stdnets, self.nets):
			stdout, _attn = snet(stdout, mask)
			out = net(out, _attn)

		if self.out_normer is not None:
			out, stdout = self.out_normer(out), self.out_normer(stdout)

		return out, stdout

	def load_base(self, base_encoder):

		self.drop = base_encoder.drop

		self.wemb = base_encoder.wemb

		self.pemb = base_encoder.pemb

		for net, snet, bnet in zip(self.nets, self.stdnets, base_encoder.nets):
			net.load_base(bnet)
			snet.load_base(bnet)

		self.out_normer = None if self.out_normer is None else base_encoder.out_normer
