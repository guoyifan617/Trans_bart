#encoding: utf-8

import sys

from utils.fmt.base import sys_open

def normd(din):

	_t = 0.0
	for k, v in din.items():
		_t += v
	rsd = {}
	for k, v in din.items():
		rsd[k] = v / _t * 100.0

	return rsd

def formatd(din):

	vl = list(din.values())
	vl.sort(reverse=True)

	td = {}
	for k, v in din.items():
		if v in td:
			td[v].append(k)
		else:
			td[v] = [k]

	rs = []

	for vu in vl:
		stv = str(vu)
		for k in td[vu]:
			rs.append(k+":"+stv)

	return rs

def handle(tagf, wf, rsf):

	rsd = {}
	rst = {}
	with sys_open(tagf, "rb") as frd, sys_open(wf, "rb") as frw:
		for lt, lw in zip(frd, frw):
			tlt, tlw = lt.strip(), lw.strip()
			if tlt and tlw:
				tlt, tlw = tlt.decode("utf-8").split(), eval(tlw.decode("utf-8"))
				# omit <sos> and <eos> since do not appear in bpe source file
				for tu, wu in zip(tlt, tlw[1:-1]):
					rst[tu] = rst.get(tu, 0) + 1
					rsd[tu] = rsd.get(tu, 0.0) + wu

	rsd_ori = formatd(normd(rsd))
	for k, v in rst.items():
		rsd[k] /= float(v)

	rsd_std = formatd(normd(rsd))
	rsd_ref = formatd(normd(rst))

	ens = "\n".encode("utf-8")
	with sys_open(rsf, "wb") as fwrt:
		fwrt.write(", ".join(rsd_std).encode("utf-8"))
		fwrt.write(ens)
		fwrt.write(", ".join(rsd_ori).encode("utf-8"))
		fwrt.write(ens)
		fwrt.write(", ".join(rsd_ref).encode("utf-8"))
		fwrt.write(ens)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3])
