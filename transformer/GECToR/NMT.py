#encoding: utf-8

from transformer.GECToR.Decoder import Decoder
from transformer.GECToR.Encoder import Encoder
from transformer.PLM.CustBERT.NMT import NMT as NMTBase
from utils.fmt.base import pad_batch
from utils.fmt.gec.gector.base import apply_op_ids
from utils.fmt.parser import parse_double_value_tuple
from utils.plm.base import set_ln_ieps
from utils.relpos.base import share_rel_pos_cache
from utils.torch.comp import torch_all_dim, torch_any_wodim

from cnfg.ihyp import *
from cnfg.vocab.gector.edit import blank_id, pad_id as edit_pad_id
from cnfg.vocab.gector.op import delete_id, keep_id
from cnfg.vocab.plm.custbert import eos_id, mask_id, pad_id

class NMT(NMTBase):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, model_name="bert", **kwargs):

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)
		enc_model_name, dec_model_name = parse_double_value_tuple(model_name)

		_ahsize = isize if ahsize is None else ahsize
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(NMT, self).__init__(isize, snwd, tnwd, (enc_layer, dec_layer,), fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=forbidden_index, model_name=model_name, **kwargs)

		self.enc = Encoder(isize, snwd, enc_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, model_name=enc_model_name)

		emb_w = self.enc.wemb.weight if global_emb else None

		self.dec = Decoder(isize, tnwd, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, emb_w=emb_w, num_head=num_head, model_name=dec_model_name)

		set_ln_ieps(self, ieps_ln_default)
		if rel_pos_enabled:
			share_rel_pos_cache(self)

	def forward(self, inpute, edit=None, mask=None, mlm_mask=None, tgt=None, prediction=False, **kwargs):

		_mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask
		_mlm_mask = inpute.eq(mask_id) if mlm_mask is None else mlm_mask
		if not torch_any_wodim(_mlm_mask).item():
			_mlm_mask = None

		return self.dec(self.enc(inpute, edit=edit, mask=_mask), mlm_mask=_mlm_mask, tgt=tgt, prediction=prediction)

	def decode(self, inpute, beam_size=1, max_len=None, length_penalty=0.0, pad_id=pad_id, edit_pad_id=edit_pad_id, delete_id=delete_id, **kwargs):

		_max_len = max(32, inpute.size(1) // 4) if max_len is None else max_len

		rs = {}
		bsize = inpute.size(0)
		_rids = list(range(bsize))
		_inpute = inpute
		_pad_mask = _inpute.eq(pad_id)
		_edit = inpute.new_full(inpute.size(), blank_id).masked_fill_(_pad_mask, edit_pad_id)
		_last_step = _max_len - 1
		for step in range(_max_len):
			if step > 0:
				_mlm_mask = _inpute.eq(mask_id)
				if not torch_any_wodim(_mlm_mask).item():
					_mlm_mask = None
			else:
				_mlm_mask = None
			_tag_out = self.dec(self.enc(_inpute, edit=_edit, mask=_pad_mask.unsqueeze(1)), mlm_mask=_mlm_mask, tgt=None, prediction=True)[-1]
			_keep_mask = _tag_out.eq(keep_id) | _inpute.eq(eos_id)
			done_trans = torch_all_dim(_keep_mask | _tag_out.eq(delete_id) | _pad_mask, -1)
			if torch_any_wodim(done_trans).item():
				_c = []
				for i, (_d, _ind, _i, _k) in enumerate(zip(done_trans.tolist(), _rids, _inpute.unbind(0), _keep_mask.unbind(0))):
					if _d:
						rs[_ind] = _i[_k]
						_c.append(i)
				for _ in reversed(_c):
					del _rids[_]
				if not _rids:
					break
				_ = ~done_trans
				_inpute = _inpute[_]
				_tag_out = _tag_out[_]
				if step < _last_step:
					_edit = _edit[_]
				else:
					break
			if step < _last_step:
				_next_i, _next_e, _mlen = [], [], 0
				for _i, _e, _t in zip(_inpute.tolist(), _edit.tolist(), _tag_out.tolist()):
					_iu, _eu = apply_op_ids(_i.tolist(), _e.tolist(), _t.tolist())
					_next_i.append(_iu)
					_next_e.append(_eu)
					_l = len(_iu)
					if _l > _mlen:
						_mlen = _l
				_inpute, _edit = _inpute.new_tensor(pad_batch(_next_i, _mlen, pad_id=pad_id)), _inpute.new_tensor(pad_batch(_next_e, _mlen, pad_id=edit_pad_id))
		if _rids:
			_mlm_mask = _inpute.eq(mask_id)
			if torch_any_wodim(_mlm_mask).item():
				_inpute[_mlm_mask] = _tag_out[_mlm_mask]
			for _ind, _i, _k in zip(_rids, _inpute.unbind(0), _inpute.ne(delete_id).unbind(0)):
				rs[_ind] = _i[_k]

		return [rs[_] for _ in range(bsize)]

	def build_task_model(self, *args, **kwargs):

		if hasattr(self.enc, "build_task_model"):
			self.enc.build_task_model(*args, **kwargs)
		if hasattr(self.dec, "build_task_model"):
			self.dec.build_task_model(*args, **kwargs)
