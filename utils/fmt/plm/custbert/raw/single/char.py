#encoding: utf-8

eos_token = "<eos>"

def sent_file_reader(sfile, max_len=510):

	_max_len = max_len + 1
	for line in sfile:
		tmp = line.strip()
		if tmp:
			tmp = tmp.decode("utf-8")
			if len(tmp) < _max_len:
				yield tuple(tmp)

def doc_file_reader(dfile, max_len=510, eos_token=eos_token):

	prev_sent, prev_sent_len = None, 0
	_max_len_s = max_len + 1
	_max_len_p = _max_len_s if eos_token is None else max_len
	for line in dfile:
		tmp = line.strip()
		if tmp:
			tmp = tmp.decode("utf-8")
			_cur_l = len(tmp)
			if _cur_l < _max_len_s:
				if prev_sent is None:
					prev_sent, prev_sent_len = tmp, _cur_l
				else:
					if (prev_sent_len + _cur_l) < _max_len_p:
						_rs = list(prev_sent)
						if eos_token is not None:
							_rs.append(eos_token)
						_rs.extend(list(tmp))
						yield tuple(_rs)
						prev_sent, prev_sent_len = None, 0
					else:
						yield tuple(prev_sent)
						prev_sent, prev_sent_len = tmp, _cur_l
			else:
				if prev_sent is not None:
					yield tuple(prev_sent)
					prev_sent, prev_sent_len = None, 0
		else:
			if prev_sent is not None:
				yield tuple(prev_sent)
				prev_sent, prev_sent_len = None, 0
	if prev_sent is not None:
		yield tuple(prev_sent)
		prev_sent, prev_sent_len = None, 0
