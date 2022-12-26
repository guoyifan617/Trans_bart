#encoding: utf-8

from utils.fmt.diff import seq_diff_reorder_insert as diff_func

#from cnfg.vocab.plm.custbert import mask_id

def generate_iter_data(src, tgt, mask_token="<mask>"):

	_diff = diff_func(src, tgt)
	_src, _tgt, _il = [], [], []
	_prev_op = None
	_iu = []
	for _op, _token in _diff:
		if _op == "i":
			_iu.append(_token)
			if _prev_op == "k":
				_tgt[-1] = "a"
		else:
			if _iu:
				_il.append(_iu)
				_iu = []
			_src.append(_token)
			_tgt.append(_op)
		_prev_op = _op
	if _iu:
		_il.append(_iu)
		_iu = None
	_edit = ["b" for _ in range(len(_src))]
	yield _src, _edit, _tgt
	_handle_mask = True if _il else False
	while _il:
		_s, _e, _t, _na = [], [], [], []
		_ril = 0
		for _su, _eu, _tu in zip(_src, _edit, _tgt):
			if _tu == "a":
				_s.append(_su)
				_s.append(mask_token)
				_e.append("k" if _eu == "b" else _eu)
				_e.append("m")
				_t.append("k")
				_ = _il[_ril]
				_ril += 1
				_t.append(_.pop(0))
				_na.append(False)
				_na.append(True if _ else False)
			else:
				_s.append(_su)
				_e.append(_tu if _eu == "b" else _eu)
				_t.append(_tu)
				_na.append(False)
		_cid = []
		for _ind, _ in enumerate(_il):
			if not _:
				_cid.append(_ind)
		for _ in reversed(_cid):
			del _il[_]
		_src, _edit, _tgt = _s, _e, _t
		yield _src, _edit, _tgt
		if any(_na):
			_s, _e, _t = [], [], []
			for _su, _eu, _tu, _au in zip(_src, _edit, _tgt, _na):
				_fill_mask = _su == mask_token
				_s.append(_tu if _fill_mask else _su)
				_e.append("i" if _fill_mask else _eu)
				_t.append("a" if _au else ("k" if _fill_mask else _tu))
			_src, _edit, _tgt = _s, _e, _t
			yield _src, _edit, _tgt
	if _handle_mask:
		_s, _e, _t = [], [], []
		for _su, _eu, _tu in zip(_src, _edit, _tgt):
			_fill_mask = _su == mask_token
			_s.append(_tu if _fill_mask else _su)
			_e.append("i" if _fill_mask else _eu)
			_t.append("k" if _fill_mask else _tu)
		_src, _edit, _tgt = _s, _e, _t
		yield _src, _edit, _tgt
