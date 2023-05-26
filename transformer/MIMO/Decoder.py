#encoding: utf-8

import torch
from math import ceil, sqrt

from modules.base import Linear
from transformer.Decoder import Decoder as DecoderBase
from utils.base import index_tensors, select_zero_
from utils.decode.beam import expand_bsize_for_beam
from utils.fmt.parser import parse_none
from utils.sampler import SampleMax
from utils.torch.comp import all_done

from cnfg.ihyp import *
from cnfg.vocab.base import eos_id, pad_id

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=True, forbidden_index=None, nmimo=4, enable_proj_bias=enable_proj_bias_default, **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, **kwargs)

		self.trans_i, self.trans_o = Linear(isize, isize * nmimo, bias=enable_proj_bias), Linear(isize, isize * nmimo, bias=enable_proj_bias)
		self.nmimo = nmimo

	# out: (bsize, nmimo, seql, isize)
	def forward(self, inpute, inputo, src_pad_mask=None, ind=None, **kwargs):

		nquery = inputo.size(-1)

		out = self.wemb(inputo)

		if self.pemb is not None:
			out = self.pemb(inputo, expand=False).add(out, alpha=sqrt(out.size(-1)))
		if self.drop is not None:
			out = self.drop(out)

		bsize, seql, isize = out.size()
		out = self.trans_i(out).view(bsize, seql, isize, self.nmimo)

		_mimo = self.training or (ind is not None)

		if _mimo:
			_out = []
			for _ind, _ou in zip([None] + ind, out.unbind(-1)):
				if _ind is None:
					_out.append(_ou)
				else:
					_out.append(_ou.index_select(0, _ind))
			out = torch.stack(_out, -1)
		out = out.sum(-1)

		_mask = self._get_subsequent_mask(nquery)

		for net in self.nets:
			out = net(inpute, out, src_pad_mask, _mask)

		if self.out_normer is not None:
			out = self.out_normer(out)

		if _mimo:
			out = self.trans_o(out).view(bsize, seql, self.nmimo, isize).transpose(1, 2).contiguous()
			out = self.lsm(self.classifier(out))
		else:
			out = self.classifier(self.trans_o(out).view(bsize, seql, self.nmimo, isize)).softmax(-1).mean(-2).log()

		return out

	def decode(self, inpute, src_pad_mask=None, beam_size=1, max_len=512, length_penalty=0.0, fill_pad=False, ensemble_decoding=False, bsize=None, **kwargs):

		return self.beam_decode(inpute, src_pad_mask, beam_size, max_len, length_penalty, fill_pad=fill_pad, ensemble_decoding=ensemble_decoding, bsize=bsize, **kwargs) if beam_size > 1 else self.greedy_decode(inpute, src_pad_mask, max_len, fill_pad=fill_pad, ensemble_decoding=ensemble_decoding, bsize=bsize, **kwargs)

	def greedy_decode(self, inpute, src_pad_mask=None, max_len=512, fill_pad=False, sample=False, ensemble_decoding=False, bsize=None, **kwargs):

		bsize = inpute.size(0) if bsize is None else bsize

		out = self.get_sos_emb(inpute, bsize=bsize)

		if self.pemb is not None:
			sqrt_isize = sqrt(out.size(-1))
			out = self.pemb.get_pos(0).add(out, alpha=sqrt_isize)

		if self.drop is not None:
			out = self.drop(out)

		seql, isize = out.size()[1:]
		if ensemble_decoding:
			out = self.trans_i(out).view(bsize, seql, isize, self.nmimo).sum(-1)
			_reduced_bsize = bsize
		else:
			_reduced_bsize = ceil(float(bsize) / float(self.nmimo))
			_er_bsize = _reduced_bsize * self.nmimo
			_npad = _er_bsize - bsize
			_bpad = out.new_zeros((_npad, seql, isize,)) if _npad > 0 else None
			_w_trans_i, _b_trans_i = self.trans_i.weight.view(isize, self.nmimo, isize).permute(1, 2, 0).contiguous(), None if self.trans_i.bias is None else self.trans_i.bias.view(isize, self.nmimo).transpose(0, 1).contiguous()
			if _bpad is not None:
				out = torch.cat((out, _bpad,), dim=0)
			out = out.view(_reduced_bsize, self.nmimo, seql, isize)
			out = out.transpose(0, 1).contiguous().view(self.nmimo, _reduced_bsize * seql, isize).bmm(_w_trans_i).transpose(0, 1).contiguous().view(_reduced_bsize, seql, self.nmimo, isize)
			if _b_trans_i is not None:
				out = out + _b_trans_i
			out = out.sum(-2)

		states = {}

		for _tmp, net in enumerate(self.nets):
			out, _state = net(inpute, (None, None,), src_pad_mask, None, out)
			states[_tmp] = _state

		if self.out_normer is not None:
			out = self.out_normer(out)

		out = self.trans_o(out).view(_reduced_bsize, seql, self.nmimo, isize)

		out = self.classifier(out).softmax(-1)
		if ensemble_decoding:
			wds = SampleMax(out.mean(-2), dim=-1, keepdim=False) if sample else out.sum(-2).argmax(dim=-1)
		else:
			out = out.transpose(1, 2).contiguous().view(_er_bsize, seql, -1).narrow(0, 0, bsize)
			wds = SampleMax(out, dim=-1, keepdim=False) if sample else out.argmax(dim=-1)

		trans = [wds]

		done_trans = wds.eq(eos_id)

		for i in range(1, max_len):

			out = self.wemb(wds)
			if self.pemb is not None:
				out = self.pemb.get_pos(i).add(out, alpha=sqrt_isize)
			if self.drop is not None:
				out = self.drop(out)

			if ensemble_decoding:
				out = self.trans_i(out).view(bsize, seql, isize, self.nmimo).sum(-1)
			else:
				if _bpad is not None:
					out = torch.cat((out, _bpad,), dim=0)
				out = out.view(_reduced_bsize, self.nmimo, seql, isize)
				out = out.transpose(0, 1).contiguous().view(self.nmimo, _reduced_bsize * seql, isize).bmm(_w_trans_i).transpose(0, 1).contiguous().view(_reduced_bsize, seql, self.nmimo, isize)
				if _b_trans_i is not None:
					out = out + _b_trans_i
				out = out.sum(-2)

			for _tmp, net in enumerate(self.nets):
				out, _state = net(inpute, states[_tmp], src_pad_mask, None, out)
				states[_tmp] = _state

			if self.out_normer is not None:
				out = self.out_normer(out)

			out = self.trans_o(out).view(_reduced_bsize, seql, self.nmimo, isize)

			out = self.classifier(out).softmax(-1)
			if ensemble_decoding:
				wds = SampleMax(out.mean(-2), dim=-1, keepdim=False) if sample else out.sum(-2).argmax(dim=-1)
			else:
				out = out.transpose(1, 2).contiguous().view(_er_bsize, seql, -1).narrow(0, 0, bsize)
				wds = SampleMax(out, dim=-1, keepdim=False) if sample else out.argmax(dim=-1)

			trans.append(wds.masked_fill(done_trans, pad_id) if fill_pad else wds)

			done_trans = done_trans | wds.eq(eos_id)
			if all_done(done_trans, bsize):
				break

		return torch.cat(trans, 1)

	def beam_decode(self, inpute, src_pad_mask=None, beam_size=8, max_len=512, length_penalty=0.0, return_all=False, clip_beam=clip_beam_with_lp, fill_pad=False, ensemble_decoding=False, bsize=None, **kwargs):

		seql = inpute.size(1)
		_ibsize = inpute.size(0)
		bsize = _ibsize if bsize is None else bsize
		_real_ibsize = _ibsize * beam_size

		beam_size2 = beam_size * beam_size
		bsizeb2 = bsize * beam_size2
		real_bsize = bsize * beam_size

		out = self.get_sos_emb(inpute, bsize=bsize)

		if length_penalty > 0.0:
			lpv = out.new_ones(real_bsize, 1)
			lpv_base = 6.0 ** length_penalty

		if self.pemb is not None:
			sqrt_isize = sqrt(out.size(-1))
			out = self.pemb.get_pos(0).add(out, alpha=sqrt_isize)

		if self.drop is not None:
			out = self.drop(out)

		nquery = out.size(1)
		if ensemble_decoding:
			out = self.trans_i(out).view(bsize, nquery, isize, self.nmimo).sum(-1)
			_reduced_bsize, _bpad = bsize, None
		else:
			_reduced_bsize = ceil(float(bsize) / float(self.nmimo))
			_er_bsize = _reduced_bsize * self.nmimo
			_npad = _er_bsize - bsize
			_bpad = out.new_zeros((_npad, nquery, isize,)) if _npad > 0 else None
			_w_trans_i, _b_trans_i = self.trans_i.weight.view(isize, self.nmimo, isize).permute(1, 2, 0).contiguous(), None if self.trans_i.bias is None else self.trans_i.bias.view(isize, self.nmimo).transpose(0, 1).contiguous()
			if _bpad is not None:
				out = torch.cat((out, _bpad,), dim=0)
			out = out.view(_reduced_bsize, self.nmimo, nquery, isize)
			out = out.transpose(0, 1).contiguous().view(self.nmimo, _reduced_bsize * nquery, isize).bmm(_w_trans_i).transpose(0, 1).contiguous().view(_reduced_bsize, nquery, self.nmimo, isize)
			if _b_trans_i is not None:
				out = out + _b_trans_i
			out = out.sum(-2)

		states = {}

		for _tmp, net in enumerate(self.nets):
			out, _state = net(inpute, (None, None,), src_pad_mask, None, out)
			states[_tmp] = _state

		if self.out_normer is not None:
			out = self.out_normer(out)

		out = self.classifier(self.trans_o(out).view(_reduced_bsize, nquery, self.nmimo, isize))

		out = out.softmax(-1).mean(-2).log() if ensemble_decoding else self.lsm(out.transpose(1, 2).contiguous().view(_er_bsize, nquery, -1).narrow(0, 0, bsize))

		scores, wds = out.topk(beam_size, dim=-1)
		scores = scores.squeeze(1)
		sum_scores = scores
		wds = wds.view(real_bsize, 1)
		trans = wds
		_inds_add_beam2 = torch.arange(0, bsizeb2, beam_size2, dtype=wds.dtype, device=wds.device).unsqueeze(1).expand(bsize, beam_size)
		_inds_add_beam = torch.arange(0, real_bsize, beam_size, dtype=wds.dtype, device=wds.device).unsqueeze(1).expand(bsize, beam_size)

		done_trans = wds.view(bsize, beam_size).eq(eos_id)

		self.repeat_cross_attn_buffer(beam_size)

		_src_pad_mask = None if src_pad_mask is None else src_pad_mask.repeat(1, beam_size, 1).view(_real_ibsize, 1, seql)

		# following 2 lines are inconsistent after commit 6fe3189ea4ad04f635422b387f00b60776df5649
		#for key, value in states.items():
		#	states[key] = repeat_bsize_for_beam_tensor(value if _bpad is None else torch.cat((value.view(bsize, -1, isize), _bpad,), dim=0), beam_size)
		states = expand_bsize_for_beam(states, beam_size=beam_size)

		if (not ensemble_decoding) and (_bpad is not None):
			_bpad = out.new_zeros((_npad, beam_size, nquery, isize,)) if _npad > 0 else None

		for step in range(1, max_len):

			out = self.wemb(wds)
			if self.pemb is not None:
				out = self.pemb.get_pos(step).add(out, alpha=sqrt_isize)
			if self.drop is not None:
				out = self.drop(out)

			if ensemble_decoding:
				out = self.trans_i(out).view(real_bsize, nquery, isize, self.nmimo).sum(-1)
			else:
				if _bpad is not None:
					out = torch.cat((out.view(bsize, beam_size, nquery, isize), _bpad,), dim=0)
				out = out.view(_reduced_bsize, self.nmimo, beam_size, nquery, isize)
				out = out.transpose(0, 1).contiguous().view(self.nmimo, _reduced_bsize * beam_size * nquery, isize).bmm(_w_trans_i).transpose(0, 1).contiguous().view(_reduced_bsize * beam_size, nquery, self.nmimo, isize)
				if _b_trans_i is not None:
					out = out + _b_trans_i
				out = out.sum(-2)

			for _tmp, net in enumerate(self.nets):
				out, _state = net(inpute, states[_tmp], _src_pad_mask, None, out)
				states[_tmp] = _state

			if self.out_normer is not None:
				out = self.out_normer(out)

			out = self.classifier(self.trans_o(out).view(_reduced_bsize, beam_size, self.nmimo, isize))

			out = out.softmax(-1).mean(-2).log() if ensemble_decoding else self.lsm(out.transpose(1, 2).contiguous().view(_er_bsize, beam_size, -1).narrow(0, 0, bsize))

			_scores, _wds = out.topk(beam_size, dim=-1)
			_done_trans_unsqueeze = done_trans.unsqueeze(2)
			_scores = (_scores.masked_fill(_done_trans_unsqueeze.expand(bsize, beam_size, beam_size), 0.0) + sum_scores.unsqueeze(2).repeat(1, 1, beam_size).masked_fill_(select_zero_(_done_trans_unsqueeze.repeat(1, 1, beam_size), -1, 0), -inf_default))

			if length_penalty > 0.0:
				lpv.masked_fill_(~done_trans.view(real_bsize, 1), ((step + 6.0) ** length_penalty) / lpv_base)

			if clip_beam and (length_penalty > 0.0):
				scores, _inds = (_scores.view(real_bsize, beam_size) / lpv.expand(real_bsize, beam_size)).view(bsize, beam_size2).topk(beam_size, dim=-1)
				_tinds = (_inds + _inds_add_beam2).view(real_bsize)
				sum_scores = _scores.view(bsizeb2).index_select(0, _tinds).view(bsize, beam_size)
			else:
				scores, _inds = _scores.view(bsize, beam_size2).topk(beam_size, dim=-1)
				_tinds = (_inds + _inds_add_beam2).view(real_bsize)
				sum_scores = scores

			wds = _wds.view(bsizeb2).index_select(0, _tinds).view(real_bsize, 1)

			_inds = (_inds // beam_size + _inds_add_beam).view(real_bsize)

			trans = torch.cat((trans.index_select(0, _inds), wds.masked_fill(done_trans.view(real_bsize, 1), pad_id) if fill_pad else wds), 1)

			done_trans = (done_trans.view(real_bsize).index_select(0, _inds) | wds.eq(eos_id).squeeze(1)).view(bsize, beam_size)

			_done = False
			if length_penalty > 0.0:
				lpv = lpv.index_select(0, _inds)
			elif (not return_all) and all_done(done_trans.select(1, 0), bsize):
				_done = True

			if _done or all_done(done_trans, real_bsize):
				break

			# ERROR: inconsistent states and inds
			states = index_tensors(states, indices=_inds, dim=0)

		if (not clip_beam) and (length_penalty > 0.0):
			scores = scores / lpv.view(bsize, beam_size)
			scores, _inds = scores.topk(beam_size, dim=-1)
			_inds = (_inds + _inds_add_beam).view(real_bsize)
			trans = trans.view(real_bsize, -1).index_select(0, _inds)

		if return_all:

			return trans.view(bsize, beam_size, -1), scores
		else:

			return trans.view(bsize, beam_size, -1).select(1, 0)
