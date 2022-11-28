#encoding: utf-8

import sys

from utils.fmt.base import ldvocab_list
from utils.fmt.doc.base import legal_vocab

# vratio: percentages of vocabulary size of retrieved words of least frequencies
# dratio: a datum will be dropped who contains high frequency words less than this ratio

def handle(srcfs, tgtfs, vcbfs, vratio, dratio=None):

	_dratio = vratio if dratio is None else dratio

	ens = "\n\n".encode("utf-8")

	vcbs, nvs = ldvocab_list(vcbfs)
	ilgs = set(vcbs[int(float(nvs) * (1.0 - vratio)):])

	with open(srcfs, "rb") as fs, open(tgtfs, "wb") as fsw:
		total = keep = 0
		cache = []
		for ls in fs:
			ls = ls.strip()
			if ls:
				cache.append(ls.decode("utf-8"))
			else:
				if cache and legal_vocab(cache, ilgs, _dratio):
					fsw.write("\n".join(cache).encode("utf-8"))
					fsw.write(ens)
					cache = []
					keep += 1
				total += 1
		if cache:
			if legal_vocab(cache, ilgs, _dratio):
				fsw.write("\n".join(cache).encode("utf-8"))
				fsw.write(ens)
				keep += 1
			total += 1
		print("%d in %d data keeped with ratio %.2f" % (keep, total, float(keep) / float(total) * 100.0 if total > 0 else 0.0))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]))
