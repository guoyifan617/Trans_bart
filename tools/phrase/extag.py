#encoding: utf-8

import sys

from utils.fmt.base import sys_open

def extract_tokens(strin):

	tmp = strin
	rind = tmp.find(")")
	rs = []
	while rind >= 0:
		lind = tmp[:rind].rfind("(")
		if lind >= 0:
			_tag, _t = tmp[lind + 1:rind].split()
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
			rs.append((_t, _tag,))
			tmp = tmp[rind + 1:]
			lind = tmp.find("(")
			if lind >= 0:
				tmp = tmp[lind:]
				rind = tmp.find(")")
			else:
				break

	return rs

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

# lparser: [[(tok, tag), ...] ...]
# lbpe: [tok, ...]
def construct_bpelist(lparser, lbpe):

	# bpel: iter([(nchar, text)])
	bpel = iter(update_bpelist(lbpe))
	rs = []
	len_tok = 0
	exception_break = False
	for tok, tag in lparser:
		len_tok += len(tok)

		while len_tok > 0:
			try:
				_nt, _ck = next(bpel)
				rs.append(tag)
				len_tok -= _nt
			except Exception as e:
				print("Inconsistent length:", lbpe, lparser)
				_exception_break = True
				break
			# comment for performance in the future if no error found in processing the training set.
			if len_tok < 0:
				print("Alignment error:", lbpe, lparser)
		if exception_break:
			break

	return rs

def process_cache(lin, bpesrc):

	rl = construct_bpelist(lin, bpesrc.split())

	return " ".join(rl)

def handle(srcf, bpef, rsf):

	ens = "\n".encode("utf-8")
	cache = []
	with sys_open(srcf, "rb") as frd, sys_open(bpef, "rb") as frb, sys_open(rsf, "wb") as fwrt:
		for line in frd:
			tmp = line.rstrip()
			if tmp:
				tmp = tmp.decode("utf-8").strip()
				if tmp.startswith("(ROOT"):
					if cache:
						rs = process_cache(cache, frb.readline().strip().decode("utf-8"))
						fwrt.write(rs.encode("utf-8"))
						fwrt.write(ens)
					cache = []
				elif tmp.endswith(")"):
					cache.extend(extract_tokens(tmp))
		if cache:
			rs = process_cache(cache, frb.readline().strip().decode("utf-8"))
			fwrt.write(rs.encode("utf-8"))
			fwrt.write(ens)
			cache = []

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3])
