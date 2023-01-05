#encoding: utf-8

import sys

from utils.fmt.base import clean_list, sys_open
from utils.fmt.vocab.token import ldvocab_list

def handle(srcfs, tgtfs, vcbfs):

	def no_unk(strin, vcbs):
		_no_unk = True
		for tok in clean_list(strin.split()):
			if not tok in vcbs:
				_no_unk = False
				break
		return _no_unk

	ens = "\n".encode("utf-8")

	vcbs, _ = ldvocab_list(vcbfs)
	vcbs = set(vcbs)

	with sys_open(srcfs, "rb") as fs, sys_open(tgtfs, "wb") as fsw:
		total = keep = 0
		for ls in fs:
			ls = ls.strip()
			if ls:
				ls = ls.decode("utf-8")
				if no_unk(ls, vcbs):
					fsw.write(ls.encode("utf-8"))
					fsw.write(ens)
					keep += 1
				total += 1
		print("%d in %d data keeped with ratio %.2f" % (keep, total, float(keep) / float(total) * 100.0 if total > 0 else 0.0))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3])
