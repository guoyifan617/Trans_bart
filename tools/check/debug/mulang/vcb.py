#encoding: utf-8

import sys
from math import acos, pi, sqrt

from utils.fmt.vocab.token import ldvocab

def lang_distance(lang1, lang2):

	common = 0.0
	l1norm = 0.0
	for k, v in lang1.items():
		_v = float(v)
		if k in lang2:
			common += _v * float(lang2[k])
		l1norm += _v * _v
	l2norm = 0.0
	for v in lang2.values():
		_v = float(v)
		l2norm += _v * _v
	if l1norm == 0.0:
		l1norm = 1.0
	if l2norm == 0.0:
		l2norm = 1.0
	cosin = sqrt(common/l1norm/l2norm)
	angle = acos(cosin) * 180.0 / pi

	return angle

def handle(srcfl, vcbf, rsf, rsxf):

	vcb, _ = ldvocab(vcbf, minf=None, omit_vsize=64000, vanilla=True)
	fvcb = {}
	ndata = {}
	for srcf, tgtf in zip(srcfl[0::2], srcfl[1::2]):
		with open(srcf, "rb") as fsrc, open(tgtf, "rb") as ftgt:
			for lsrc, ltgt in zip(fsrc, ftgt):
				tsrc, ttgt = lsrc.strip(), ltgt.strip()
				if tsrc and ttgt:
					task = tsrc.decode("utf-8").split()[0]
					ndata[task] = ndata.get(task, 0) + 1
					if task not in fvcb:
						fvcb[task] = {}
					tdict = fvcb[task]
					for token in ttgt.decode("utf-8").split():
						if token:
							if token in vcb:
								tdict[token] = tdict.get(token, 0) + 1

	distance = {}
	for k, v in fvcb.items():
		for l2k, l2v in fvcb.items():
			if k != l2k:
				if k in distance.get(l2k, {}):
					rs = distance[l2k][k]
				else:
					rs = lang_distance(v, l2v)
				if k in distance:
					distance[k][l2k] = rs
				else:
					distance[k] = {l2k: rs}

	ens = "\n".encode("utf-8")
	with open(rsf, "wb") as f:
		f.write(repr({"ndata": ndata, "vcb": fvcb, "dist": distance}).encode("utf-8"))
		f.write(ens)

	s_ndata = {}
	for k, v in ndata.items():
		if v in s_ndata:
			s_ndata[v].append(k)
		else:
			s_ndata[v] = [k]
	tmp = list(s_ndata.keys())
	tmp.sort(reverse=True)
	t_ordered = []
	for tmpu in tmp:
		t_ordered.extend(sorted(s_ndata[tmpu]))

	doc_head = ",,".join(t_ordered + [""])
	doc_cont = []
	for slang in t_ordered:
		tmp = {}
		d = distance[slang]
		doc_l = []
		for k, v in d.items():
			if v in tmp:
				tmp[v].append(k)
			else:
				tmp[v] = [k]
		_sv = list(tmp.keys())
		_sv.sort()
		for k in _sv:
			for _vu in sorted(tmp[k]):
				doc_l.append("%s,%f" % (_vu,k,))
		doc_cont.append(doc_l)
	doc_cont = list(zip(*doc_cont))
	with open(rsxf, "wb") as f:
		f.write(doc_head.encode("utf-8"))
		f.write(ens)
		for doc_l in doc_cont:
			f.write(",".join(doc_l).encode("utf-8"))
			f.write(ens)

if __name__ == "__main__":
	handle(sys.argv[1:-3], sys.argv[-3], sys.argv[-2], sys.argv[-1])
