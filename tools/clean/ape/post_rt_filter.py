#encoding: utf-8

import sys

from utils.fmt.base import clean_list, sys_open
from utils.fmt.vocab.token import ldvocab_list

def covered(srcl, mtl, pel, srcvcb, mtvcb, tgtvcb):

	for line, vcb in zip((srcl, mtl, pel,), (srcvcb, mtvcb, tgtvcb,),):
		for tok in line:
			if not tok in vcb:
				return False
	return True

def handle(srcfs, srcfm, srcft, tgtfs, tgtfm, tgtft, vcbfs, vcbfm, vcbft, max_len=256):

	_max_len = max(1, max_len - 2)

	ens = "\n".encode("utf-8")

	vcbs, vcbm, vcbt = set(ldvocab_list(vcbfs)[0]), set(ldvocab_list(vcbfm)[0]), set(ldvocab_list(vcbft)[0])

	with sys_open(srcfs, "rb") as fs, sys_open(srcfm, "rb") as fm, sys_open(srcft, "rb") as ft, sys_open(tgtfs, "wb") as fsw, sys_open(tgtfm, "wb") as fmw, sys_open(tgtft, "wb") as ftw:
		total = keep = 0
		for ls, lm, lt in zip(fs, fm, ft):
			ls, lm, lt = ls.strip(), lm.strip(), lt.strip()
			if ls and lm and lt:
				ls, lm, lt = clean_list(ls.decode("utf-8").split()), clean_list(lm.decode("utf-8").split()), clean_list(lt.decode("utf-8").split())
				if (len(ls) <= _max_len) and (len(lm) <= _max_len) and (len(lt) <= _max_len) and covered(ls, lm, lt, vcbs, vcbm, vcbt):
					fsw.write(" ".join(ls).encode("utf-8"))
					fsw.write(ens)
					fmw.write(" ".join(lm).encode("utf-8"))
					fmw.write(ens)
					ftw.write(" ".join(lt).encode("utf-8"))
					ftw.write(ens)
					keep += 1
				total += 1
		print("%d in %d data keeped with ratio %.2f" % (keep, total, float(keep) / float(total) * 100.0 if total > 0 else 0.0))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9])
