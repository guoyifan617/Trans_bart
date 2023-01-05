#encoding: utf-8

import sys
from math import ceil

from utils.fmt.base import sys_open

def adv_strip(strin):

	ct = 0
	for tmp in strin:
		if tmp == " ":
			ct += 1
		else:
			break
	return ct // 2, strin.strip()

def extract_tokens(strin):

	tmp = strin
	rind = tmp.find(")")
	rs = []
	while rind >= 0:
		lind = tmp[:rind].rfind("(")
		if lind >= 0:
			_t = tmp[lind + 1:rind].split()[-1]
			if _t == "-LRB-":
				_t = "("
			elif _t == "-RRB-":
				_t = ")"
			elif _t == "-LSB-":
				_t = "["
			elif _t == "-RSB-":
				_t = "]"
			elif _t == "``" or _t == "\"\"":
				_t = "\""
			rs.append(_t)
			tmp = tmp[rind + 1:]
			lind = tmp.find("(")
			if lind >= 0:
				tmp = tmp[lind:]
				rind = tmp.find(")")
			else:
				break

	return len(rs), rs

def reach_upper(find, prep_data, sid, remain_tokens, slevel):

	id_parent = id_brother = False
	curid = sid
	rs = [prep_data]
	_plevel = slevel
	_rtk = remain_tokens
	while _rtk > 0 and curid >= 0:
		_curlevel, _curlen, _curdata = find[curid]
		if _rtk >= _curlen:
			if _curlevel <= _plevel:
				id_parent = curid
				_plevel = _curlevel
			elif _curlevel == slevel:
				id_brother = curid
			rs.append(_curdata)
			_rtk -= _curlen
			curid -= 1
		else:
			break

	if id_parent:
		trs = rs[:sid - id_parent + 2]
		trs.reverse()
		trs = " ".join(trs)
	elif id_brother:
		trs = rs[:sid - id_brother + 2]
		trs.reverse()
		trs = " ".join(trs)
	else:
		trs = None

	return id_parent, id_brother, trs

def update_bpelist(lbpe):

	rs = []
	for tmp in lbpe:
		if tmp.endswith("@@"):
			rs.append((len(tmp) - 2, tmp,))
		elif tmp == "@-@":
			rs.append((1, tmp,))
		else:
			rs.append((len(tmp), tmp,))

	return rs

# lparser: [(level, ntok, [tok, ...]) ...]
# lbpe: [tok, ...]
def construct_bpelist(lparser, lbpe):

	# bpel: iter([(nchar, text)])
	bpel = iter(update_bpelist(lbpe))
	rs = []
	len_tok = 0
	exception_break = False
	for level, ntok, tokl in lparser:
		tmp = []
		for tok in tokl:
			len_tok += len(tok)
		#print(len_tok)
		while len_tok > 0:
			try:
				_nt, _ck = next(bpel)
				tmp.append(_ck)
				len_tok -= _nt
			except Exception as e:
				print("Inconsistent length:", lbpe, lparser)
				_exception_break = True
				break
			# comment for performance in the future if no error found in processing the training set.
			if len_tok < 0:
				print("Alignment error:", lbpe, lparser, tmp)
		if tmp:
			rs.append((level, len(tmp), " ".join(tmp)))
		if exception_break:
			break

	return rs

def process_cache(lin, bpesrc, max_chunk_tokens=8, min_chunks=8):

	rl = construct_bpelist(lin, bpesrc.split())

	seql = 0
	# build fast indexing dict, fbind: {list_index: (level, ntok, text) ...}
	find = {}
	for ind, tmp in enumerate(rl):
		find[ind] = tmp
		seql += tmp[1]

	mxtok = max(min(max_chunk_tokens, ceil(seql / min_chunks)), 2)

	rs = []
	curid = len(rl) - 1
	while curid >= 0:
		_curlevel, _curlen, _curdata = find[curid]
		if _curlen >= mxtok:
			rs.append(_curdata)
			curid -= 1
		elif curid == 0:
			rs.append(_curdata)
			curid -= 1
		else:
			_id_parent, _id_brother, _trs = reach_upper(find, _curdata, curid - 1, mxtok - _curlen, _curlevel)
			if _id_parent:
				rs.append(_trs)
				curid = _id_parent - 1
			elif _id_brother:
				rs.append(_trs)
				curid = _id_brother - 1
			else:
				rs.append(_curdata)
				curid -= 1

	rs.reverse()

	return "\t".join(rs)

def handle(srcf, bpef, rsf, max_phrase_size=8, min_phrases=8):

	ens = "\n".encode("utf-8")
	cache = []
	with sys_open(srcf, "rb") as frd, sys_open(bpef, "rb") as frb, sys_open(rsf, "wb") as fwrt:
		for line in frd:
			tmp = line.rstrip()
			if tmp:
				level, tmp = adv_strip(tmp.decode("utf-8"))
				if tmp.startswith("(ROOT"):
					if cache:
						rs = process_cache(cache, frb.readline().strip().decode("utf-8"), max_phrase_size, min_phrases)
						fwrt.write(rs.encode("utf-8"))
						fwrt.write(ens)
					cache = []
				elif tmp.endswith(")"):
					cache.append((level, *extract_tokens(tmp),))
		if cache:
			rs = process_cache(cache, frb.readline().strip().decode("utf-8"), max_phrase_size, min_phrases)
			fwrt.write(rs.encode("utf-8"))
			fwrt.write(ens)
			cache = []

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3])
