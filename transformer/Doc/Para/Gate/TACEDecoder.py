#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.base import Dropout
from transformer.Doc.Para.Gate.CEDecoder import Decoder as DecoderBase
from utils.base import index_tensors, select_zero_
from utils.decode.beam import expand_bsize_for_beam
from utils.doc.paragate.base4torch import clear_pad_mask
from utils.sampler import SampleMax
from utils.torch.comp import all_done

from cnfg.ihyp import *
from cnfg.vocab.base import eos_id, pad_id

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=True, forbidden_index=None, nprev_context=2, num_layer_cross=1, drop_tok=None, **kwargs):

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, nprev_context=nprev_context, num_layer_cross=num_layer_cross, drop_tok=drop_tok, **kwargs)

		self.nlo = num_layer + 1
		self.tattn_w = nn.Parameter(torch.Tensor(self.nlo).uniform_(- sqrt(1.0 / self.nlo), sqrt(1.0 / self.nlo)))
		self.tattn_drop = Dropout(dropout) if dropout > 0.0 else None

	# inpute: (bsize * nsent, seql, isize)
	# inputo: (bsize, nsent, nquery)
	# inputot: shifted inputo with noise
	# enc_context: (bsize * (nsent - 1), _seql, isize)
	# context: (bsize * (nsent - 1), _2seql, isize)
	# inputot_mask: (bsize, 1, nsent, nquery), get by:
	#	inputot.eq(pad_id).unsqueeze(1)
	def forward(self, inpute, inputo, inputot, enc_context, context, src_pad_mask=None, context_mask=None, inputot_mask=None, enc_context_mask=None, **kwargs):

		# forward the whole target documents
		out_emb = self.wemb(inputo)
		bsize, nsent, nquery, isize = out_emb.size()
		sbsize = bsize * nsent
		out = out_emb.view(sbsize, nquery, isize)
		if self.pemb is not None:
			out = self.pemb(out.select(-1, 0), expand=False).add(out, alpha=sqrt(out.size(-1)))

		if self.drop is not None:
			out = self.drop(out)

		outl = [out]

		_src_pad_mask = src_pad_mask.view(sbsize, 1, -1)
		_tgt_sub_mask = self._get_subsequent_mask(nquery)

		for net in self.nets:
			out = net(inpute, out, _src_pad_mask, _tgt_sub_mask)
			outl.append(out)

		if self.out_normer is not None:
			out = self.out_normer(out)
			outl[-1] = out

		# collect for inputo to inputo_context attention
		dece = out.view(bsize, nsent, nquery, isize)
		inputo_context = torch.stack(outl, -1).view(bsize, nsent, nquery, -1).narrow(1, 0, nsent - 1).contiguous().view(-1, self.nlo).mv(self.tattn_w.softmax(dim=0) if self.tattn_drop is None else self.tattn_drop(self.tattn_w).softmax(dim=0)).view(bsize, nsent - 1, nquery, isize)
		_inpute = inpute.view(bsize, nsent, -1, isize)
		inputo_context = torch.cat((inputo_context, (self.wemb(inputot) if self.tokdrop is None else self.tokdrop(torch.cat((out_emb.narrow(2, 1, nquery - 1), self.wemb(inputot.narrow(2, -1, 1)),), dim=2))).narrow(1, 0, nsent - 1),), dim=-1)
		dec_contexts = []
		dec_context_masks = []
		_isize_ctx = isize * 2
		_nsent_context_base = nsent - self.nprev_context
		for i, _sent_pemb in enumerate(self.sent_pemb.unbind(0)):
			_num_context = _nsent_context_base + i
			_context, _mask = clear_pad_mask(inputo_context.narrow(1, 0, _num_context), inputot_mask.narrow(2, 0, _num_context), dim=2, mask_dim=-1)
			_context = _context + _sent_pemb
			_num_pad = self.nprev_context - i - 1
			if _num_pad > 0:
				_seql = _context.size(2)
				_context = torch.cat((_context.new_zeros((bsize, _num_pad, _seql, _isize_ctx),), _context,), dim=1)
				_mask = torch.cat((_mask.new_ones((bsize, 1, _num_pad, _seql,),), _mask,), dim=2)
			dec_contexts.append(_context)
			dec_context_masks.append(_mask)
		dec_contexts = torch.cat(dec_contexts, dim=2)
		dec_context_masks = torch.cat(dec_context_masks, dim=-1)

		_nsent_o = nsent - 1
		out, inputot_mask = clear_pad_mask(dece.narrow(1, 1, _nsent_o), inputot_mask.narrow(2, 1, _nsent_o), dim=2, mask_dim=-1)
		_ebsize = bsize * _nsent_o
		nquery = out.size(2)
		out, dec_contexts, dec_context_masks = out.contiguous().view(_ebsize, nquery, isize), dec_contexts.view(_ebsize, -1, _isize_ctx), dec_context_masks.view(_ebsize, 1, -1)
		_tgt_sub_mask = self._get_subsequent_mask(nquery)
		for net in self.gdec:
			out = net(enc_context, out, context, dec_contexts, enc_context_mask, _tgt_sub_mask, context_mask, dec_context_masks)

		if self.out_normer_ctx is not None:
			out = self.out_normer_ctx(out)

		out = self.lsm(self.classifier(out)).view(bsize, _nsent_o, nquery, -1)

		return out

	def greedy_decode(self, inpute, inputo, enc_context, context, src_pad_mask=None, context_mask=None, max_len=512, fill_pad=False, sample=False, **kwargs):

		# forward the whole target documents
		out_emb = self.wemb(inputo)
		bsize, nsent, nquery, isize = out_emb.size()
		sbsize = bsize * nsent
		lo = nquery - 1
		out = out_emb.view(sbsize, nquery, isize).narrow(1, 0, lo)

		if self.pemb is not None:
			sqrt_isize = sqrt(out.size(-1))
			out = self.pemb(out.select(-1, 0), expand=False).add(out, alpha=sqrt_isize)

		if self.drop is not None:
			out = self.drop(out)

		outl = [out]

		_src_pad_mask = src_pad_mask.view(sbsize, 1, -1)
		_tgt_sub_mask = self._get_subsequent_mask(lo)

		for net in self.nets:
			out = net(inpute, out, _src_pad_mask, _tgt_sub_mask)
			outl.append(out)

		if self.out_normer is not None:
			out = self.out_normer(out)
			outl[-1] = out

		# collect for inputo to inputo_context attention
		dece = out.view(bsize, nsent, lo, isize)
		inputo_context = torch.stack(outl, -1).view(bsize, nsent, nquery, -1).narrow(1, 0, nsent - 1).contiguous().view(-1, self.nlo).mv(self.tattn_w.softmax(dim=0) if self.tattn_drop is None else self.tattn_drop(self.tattn_w).softmax(dim=0)).view(bsize, nsent - 1, nquery, isize)
		_inpute = inpute.view(bsize, nsent, -1, isize)
		inputo_context = torch.cat((inputo_context, out_emb.narrow(1, 0, nsent - 1).narrow(2, 1, lo),), dim=-1)
		inputot_mask = inputo.narrow(1, 1, lo).eq(pad_id)
		dec_contexts = []
		dec_context_masks = []
		_isize_ctx = isize * 2
		_nsent_context_base = nsent - self.nprev_context
		for i in range(self.nprev_context):
			_num_context = _nsent_context_base + i
			_context, _mask = clear_pad_mask(inputo_context.narrow(1, 0, _num_context), inputot_mask.narrow(2, 0, _num_context), dim=2, mask_dim=-1)
			_context = _context + self.sent_pemb[i]
			_num_pad = self.nprev_context - i - 1
			if _num_pad > 0:
				_seql = _context.size(2)
				_context = torch.cat((_context.new_zeros((bsize, _num_pad, _seql, _isize_ctx),), _context,), dim=1)
				_mask = torch.cat((_mask.new_ones((bsize, 1, _num_pad, _seql,),), _mask,), dim=2)
			dec_contexts.append(_context)
			dec_context_masks.append(_mask)
		dec_contexts = torch.cat(dec_contexts, dim=2)
		dec_context_masks = torch.cat(dec_context_masks, dim=-1)

		_nsent_o = nsent - 1
		inpute, _src_pad_mask = clear_pad_mask(_inpute.narrow(1, 1, _nsent_o), src_pad_mask.narrow(2, 1, _nsent_o), dim=2, mask_dim=-1)
		bsize *= _nsent_o
		inpute, _src_pad_mask, dec_contexts, dec_context_masks = inpute.contiguous().view(bsize, -1, isize), _src_pad_mask.contiguous().view(bsize, 1, -1), dec_contexts.view(bsize, -1, _isize_ctx), dec_context_masks.view(bsize, 1, -1)

		out = self.get_sos_emb(inpute)

		if self.pemb is not None:
			out = self.pemb.get_pos(0).add(out, alpha=sqrt_isize)
		if self.drop is not None:
			out = self.drop(out)

		states = {}

		for _tmp, net in enumerate(self.nets):
			out, _state = net(inpute, (None, None,), _src_pad_mask, None, out)
			states[_tmp] = _state

		if self.out_normer is not None:
			out = self.out_normer(out)

		for _tmp, net in enumerate(self.gdec, _tmp + 1):
			out, _state = net(enc_context, None, context, dec_contexts, _src_pad_mask, None, context_mask, dec_context_masks, out)
			states[_tmp] = _state

		if self.out_normer_ctx is not None:
			out = self.out_normer_ctx(out)

		out = self.classifier(out)
		wds = SampleMax(out.softmax(-1), dim=-1, keepdim=False) if sample else out.argmax(dim=-1)

		trans = [wds]

		done_trans = wds.eq(eos_id)

		for i in range(1, max_len):

			out = self.wemb(wds)
			if self.pemb is not None:
				out = self.pemb.get_pos(i).add(out, alpha=sqrt_isize)
			if self.drop is not None:
				out = self.drop(out)

			for _tmp, net in enumerate(self.nets):
				out, _state = net(inpute, states[_tmp], _src_pad_mask, None, out)
				states[_tmp] = _state

			if self.out_normer is not None:
				out = self.out_normer(out)

			for _tmp, net in enumerate(self.gdec, _tmp + 1):
				out, _state = net(enc_context, states[_tmp], context, dec_contexts, _src_pad_mask, None, context_mask, dec_context_masks, out)
				states[_tmp] = _state

			if self.out_normer_ctx is not None:
				out = self.out_normer_ctx(out)

			out = self.classifier(out)
			wds = SampleMax(out.softmax(-1), dim=-1, keepdim=False) if sample else out.argmax(dim=-1)

			trans.append(wds.masked_fill(done_trans, pad_id) if fill_pad else wds)

			done_trans = done_trans | wds.eq(eos_id)
			if all_done(done_trans, bsize):
				break

		return torch.cat(trans, 1)

	def beam_decode(self, inpute, inputo, enc_context, context, src_pad_mask=None, context_mask=None, beam_size=8, max_len=512, length_penalty=0.0, return_all=False, clip_beam=clip_beam_with_lp, fill_pad=False, **kwargs):

		# forward the whole target documents
		out_emb = self.wemb(inputo)
		bsize, nsent, nquery, isize = out_emb.size()
		sbsize = bsize * nsent
		lo = nquery - 1
		out = out_emb.view(sbsize, nquery, isize).narrow(1, 0, lo)

		if self.pemb is not None:
			sqrt_isize = sqrt(out.size(-1))
			out = self.pemb(out.select(-1, 0), expand=False).add(out, alpha=sqrt_isize)

		if self.drop is not None:
			out = self.drop(out)

		outl = [out]

		_src_pad_mask = src_pad_mask.view(sbsize, 1, -1)
		_tgt_sub_mask = self._get_subsequent_mask(lo)

		for net in self.nets:
			out = net(inpute, out, _src_pad_mask, _tgt_sub_mask)
			outl.append(out)

		if self.out_normer is not None:
			out = self.out_normer(out)
			outl[-1] = out

		# collect for inputo to inputo_context attention
		dece = out.view(bsize, nsent, lo, isize)
		inputo_context = torch.stack(outl, -1).view(bsize, nsent, nquery, -1).narrow(1, 0, nsent - 1).contiguous().view(-1, self.nlo).mv(self.tattn_w.softmax(dim=0) if self.tattn_drop is None else self.tattn_drop(self.tattn_w).softmax(dim=0)).view(bsize, nsent - 1, nquery, isize)
		_inpute = inpute.view(bsize, nsent, -1, isize)
		inputo_context = torch.cat((inputo_context, out_emb.narrow(1, 0, nsent - 1).narrow(2, 1, lo),), dim=-1)
		inputot_mask = inputo.narrow(1, 1, lo).eq(pad_id)
		dec_contexts = []
		dec_context_masks = []
		_isize_ctx = isize * 2
		_nsent_context_base = nsent - self.nprev_context
		for i in range(self.nprev_context):
			_num_context = _nsent_context_base + i
			_context, _mask = clear_pad_mask(inputo_context.narrow(1, 0, _num_context), inputot_mask.narrow(2, 0, _num_context), dim=2, mask_dim=-1)
			_context = _context + self.sent_pemb[i]
			_num_pad = self.nprev_context - i - 1
			if _num_pad > 0:
				_seql = _context.size(2)
				_context = torch.cat((_context.new_zeros((bsize, _num_pad, _seql, _isize_ctx),), _context,), dim=1)
				_mask = torch.cat((_mask.new_ones((bsize, 1, _num_pad, _seql,),), _mask,), dim=2)
			dec_contexts.append(_context)
			dec_context_masks.append(_mask)
		dec_contexts = torch.cat(dec_contexts, dim=2)
		dec_context_masks = torch.cat(dec_context_masks, dim=-1)

		_nsent_o = nsent - 1
		inpute, _src_pad_mask = clear_pad_mask(_inpute.narrow(1, 1, _nsent_o), src_pad_mask.narrow(2, 1, _nsent_o), dim=2, mask_dim=-1)
		bsize *= _nsent_o
		inpute, _src_pad_mask, dec_contexts, dec_context_masks = inpute.contiguous().view(bsize, -1, isize), _src_pad_mask.contiguous().view(bsize, 1, -1), dec_contexts.view(bsize, -1, _isize_ctx), dec_context_masks.view(bsize, 1, -1)

		beam_size2 = beam_size * beam_size
		bsizeb2 = bsize * beam_size2
		real_bsize = bsize * beam_size

		out = self.get_sos_emb(inpute)

		if length_penalty > 0.0:
			lpv = out.new_ones(real_bsize, 1)
			lpv_base = 6.0 ** length_penalty

		if self.pemb is not None:
			out = self.pemb.get_pos(0).add(out, alpha=sqrt_isize)
		if self.drop is not None:
			out = self.drop(out)

		states = {}

		for _tmp, net in enumerate(self.nets):
			out, _state = net(inpute, (None, None,), _src_pad_mask, None, out)
			states[_tmp] = _state

		if self.out_normer is not None:
			out = self.out_normer(out)

		for _tmp, net in enumerate(self.gdec, _tmp + 1):
			out, _state = net(enc_context, None, context, dec_contexts, _src_pad_mask, None, context_mask, dec_context_masks, out)
			states[_tmp] = _state

		if self.out_normer_ctx is not None:
			out = self.out_normer_ctx(out)

		out = self.lsm(self.classifier(out))

		scores, wds = out.topk(beam_size, dim=-1)
		scores = scores.squeeze(1)
		sum_scores = scores
		wds = wds.view(real_bsize, 1)
		trans = wds
		_inds_add_beam2 = torch.arange(0, bsizeb2, beam_size2, dtype=wds.dtype, device=wds.device).unsqueeze(1).expand(bsize, beam_size)
		_inds_add_beam = torch.arange(0, real_bsize, beam_size, dtype=wds.dtype, device=wds.device).unsqueeze(1).expand(bsize, beam_size)

		done_trans = wds.view(bsize, beam_size).eq(eos_id)

		#enc_context = enc_context.repeat(1, beam_size, 1).view(real_bsize, seql, isize)
		self.repeat_cross_attn_buffer(beam_size)

		_src_pad_mask = _src_pad_mask.repeat(1, beam_size, 1).view(real_bsize, 1, seql)
		_cseql = context.size(1)
		_context = context.repeat(1, beam_size, 1).view(real_bsize, _cseql, isize)
		_context_mask = context_mask.repeat(1, beam_size, 1).view(real_bsize, 1, _cseql)

		#dec_contexts = dec_contexts.repeat(1, beam_size, 1).view(real_bsize, -1, _isize_ctx)
		dec_context_masks = dec_context_masks.repeat(1, beam_size, 1).view(real_bsize, 1, -1)

		states = expand_bsize_for_beam(states, beam_size=beam_size)

		for step in range(1, max_len):

			out = self.wemb(wds)
			if self.pemb is not None:
				out = self.pemb.get_pos(step).add(out, alpha=sqrt_isize)
			if self.drop is not None:
				out = self.drop(out)

			for _tmp, net in enumerate(self.nets):
				out, _state = net(inpute, states[_tmp], _src_pad_mask, None, out)
				states[_tmp] = _state

			if self.out_normer is not None:
				out = self.out_normer(out)

			for _tmp, net in enumerate(self.gdec, _tmp + 1):
				out, _state = net(enc_context, states[_tmp], _context, dec_contexts, _src_pad_mask, None, _context_mask, dec_context_masks, out)
				states[_tmp] = _state

			if self.out_normer_ctx is not None:
				out = self.out_normer_ctx(out)

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
