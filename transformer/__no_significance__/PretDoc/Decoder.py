#encoding: utf-8

import torch
from math import sqrt

from modules.mono import PositionalEmb
from transformer.Doc.Para.Base.Decoder import Decoder as DecoderBase
from utils.base import index_tensors, select_zero_
from utils.decode.beam import expand_bsize_for_beam
from utils.sampler import SampleMax
from utils.torch.comp import all_done

from cnfg.ihyp import *
from cnfg.vocab.mono import eos_id, pad_id

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=True, forbidden_index=None, **kwargs):

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, nprev_context=1, **kwargs)

		self.pemb = PositionalEmb(isize, xseql, 0, 0)
		self.semb = PositionalEmb(isize, xseql, 0, 0)#5000, isize or 5000, be consistent with MDoc.Decoder

	def forward(self, inpute, inputo, inputc, src_pad_mask=None, context_mask=None, start_sent_id=0, **kwargs):

		bsize, nsent, nquery = inputo.size()

		out = self.wemb(inputo)
		out = self.pemb(inputo.select(1, 0), expand=False).unsqueeze(1) + self.semb(inputo.select(-1, 0), expand=False, sind=start_sent_id).unsqueeze(2) + out * sqrt(isize)

		if self.drop is not None:
			out = self.drop(out)

		out = out.view(-1, nquery, isize)

		_src_pad_mask = None if src_pad_mask is None else src_pad_mask.view(-1, 1, src_pad_mask.size(-1))
		_mask = self._get_subsequent_mask(nquery)

		for net in self.nets:
			out = net(inpute, out, inputc, _src_pad_mask, _mask, context_mask)

		if self.out_normer is not None:
			out = self.out_normer(out)

		out = self.lsm(self.classifier(out)).view(bsize, nsent, nquery, -1)

		return out

	def greedy_decode(self, inpute, inputc, src_pad_mask=None, context_mask=None, max_len=512, fill_pad=False, sample=False, **kwargs):

		bsize = inpute.size(0)
		nsent = inputc[0].size(1)

		out = self.get_sos_emb(inpute)
		sqrt_isize = sqrt(out.size(-1))

		sent_emb = self.semb(out.new_empty((bsize // nsent, nsent), dtype=out.dtype, device=out.device)).contiguous().view(bsize, 1, isize)

		out = self.pemb.get_pos(0) + sent_emb + out * sqrt_isize

		if self.drop is not None:
			out = self.drop(out)
		out = out.view(-1, 1, isize)

		states = {}

		for _tmp, net in enumerate(self.nets):
			out, _state = net(inpute, (None, None,), inputc, src_pad_mask, context_mask, None, out)
			states[_tmp] = _state

		if self.out_normer is not None:
			out = self.out_normer(out)

		out = self.classifier(out)
		wds = SampleMax(out.softmax(-1), dim=-1, keepdim=False) if sample else out.argmax(dim=-1)

		trans = [wds]

		done_trans = wds.eq(eos_id)

		for i in range(1, max_len):

			out = self.pemb.get_pos(i) + sent_emb + self.wemb(wds) * sqrt_isize

			if self.drop is not None:
				out = self.drop(out)

			for _tmp, net in enumerate(self.nets):
				out, _state = net(inpute, states[_tmp], inputc, src_pad_mask, None, context_mask, out)
				states[_tmp] = _state

			if self.out_normer is not None:
				out = self.out_normer(out)

			out = self.classifier(out)
			wds = SampleMax(out.softmax(-1), dim=-1, keepdim=False) if sample else out.argmax(dim=-1)

			trans.append(wds.masked_fill(done_trans, pad_id) if fill_pad else wds)

			done_trans = done_trans | wds.eq(eos_id)
			if all_done(done_trans, bsize):
				break

		return torch.cat(trans, 1)

	def beam_decode(self, inpute, inputc, src_pad_mask=None, context_mask=None, beam_size=8, max_len=512, length_penalty=0.0, return_all=False, clip_beam=clip_beam_with_lp, fill_pad=False, **kwargs):

		bsize, seql = inpute.size()[:2]
		nsent = inputc[0].size(1)

		beam_size2 = beam_size * beam_size
		bsizeb2 = bsize * beam_size2
		real_bsize = bsize * beam_size

		out = self.get_sos_emb(inpute)
		sqrt_isize = sqrt(out.size(-1))

		if length_penalty > 0.0:
			lpv = out.new_ones(real_bsize, 1)
			lpv_base = 6.0 ** length_penalty

		sent_emb = self.semb(out.new_empty((bsize // nsent, nsent), dtype=out.dtype, device=out.device)).contiguous().view(bsize, 1, isize)

		out = self.pemb.get_pos(0) + sent_emb + out * sqrt_isize

		if self.drop is not None:
			out = self.drop(out)

		states = {}

		for _tmp, net in enumerate(self.nets):
			out, _state = net(inpute, (None, None,), inputc, src_pad_mask, context_mask, None, out)
			states[_tmp] = _state

		if self.out_normer is not None:
			out = self.out_normer(out)

		out = self.lsm(self.classifier(out))

		scores, wds = out.topk(beam_size, dim=-1)
		scores = scores.squeeze(1)
		sum_scores = scores
		wds = wds.view(real_bsize, 1)
		trans = wds
		_inds_add_beam2 = torch.arange(0, bsizeb2, beam_size2, dtype=wds.dtype, device=wds.device).unsqueeze(1).expand(bsize, beam_size)
		_inds_add_beam = torch.arange(0, real_bsize, beam_size, dtype=wds.dtype, device=wds.device).unsqueeze(1).expand(bsize, beam_size)

		done_trans = wds.view(bsize, beam_size).eq(eos_id)

		self.repeat_cross_attn_buffer(beam_size)

		_src_pad_mask = None if src_pad_mask is None else src_pad_mask.repeat(1, beam_size, 1).view(real_bsize, 1, seql)
		_cbsize, _cseql = inputc[0].size()[:2]
		_creal_bsize = _cbsize * beam_size
		_context_mask = None if context_mask is None else [None if cu is None else cu.repeat(1, beam_size, 1).view(_creal_bsize, 1, _cseql) for cu in context_mask]

		_inputc = [inputu.repeat(1, beam_size, 1).view(_creal_bsize, _cseql, isize) for inputu in inputc]
		sent_emb = sent_emb.repeat(1, beam_size, 1).view(-1, 1, isize)

		states = expand_bsize_for_beam(states, beam_size=beam_size)

		for step in range(1, max_len):

			out = self.pemb.get_pos(step) + sent_emb + self.wemb(wds) * sqrt_isize

			if self.drop is not None:
				out = self.drop(out)

			for _tmp, net in enumerate(self.nets):
				out, _state = net(inpute, states[_tmp], _inputc, _src_pad_mask, None, _context_mask, out)
				states[_tmp] = _state

			if self.out_normer is not None:
				out = self.out_normer(out)

			out = self.lsm(self.classifier(out)).view(bsize, beam_size, -1)

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

	def load_base(self, base_decoder):

		self.drop = base_decoder.drop

		self.wemb = base_decoder.wemb

		for snet, bnet in zip(self.nets, base_decoder.nets):
			snet.load_base(bnet)

		self.classifier = base_decoder.classifier

		self.lsm = base_decoder.lsm

		self.out_normer = None if self.out_normer is None else base_decoder.out_normer

	def get_loaded_para_models(self):

		rs = [self.wemb, self.classifier]
		if self.out_normer is not None:
			rs.append(self.out_normer)
		for net in self.nets:
			rs.extend([net.self_attn, net.cross_attn, net.ff, net.layer_normer1, net.layer_normer2])

		return rs
