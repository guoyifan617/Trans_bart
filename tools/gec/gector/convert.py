#encoding: utf-8

import sys

from utils.fmt.base import iter_to_int, iter_to_str, sys_open
from utils.fmt.gec.gector.base import generate_iter_data

def handle(srcf, tgtf, rssf, rsef, rstf):

	ens = "\n".encode("utf-8")
	with sys_open(srcf, "rb") as frds, sys_open(tgtf, "rb") as frdt, sys_open(rssf, "wb") as fwrts, sys_open(rsef, "wb") as fwrte, sys_open(rstf, "wb") as fwrtt:
		for s, t in zip(frds, frdt):
			_s, _t = s.strip(), t.strip()
			if _s and _t:
				_s, _t = tuple(iter_to_int(_s.decode("utf-8").split())), tuple(iter_to_int(_t.decode("utf-8").split()))
				for _src, _edit, _tgt in generate_iter_data(_s, _t):
					fwrts.write(" ".join(iter_to_str(_src)).encode("utf-8"))
					fwrts.write(ens)
					fwrte.write(" ".join(iter_to_str(_edit)).encode("utf-8"))
					fwrte.write(ens)
					fwrtt.write(" ".join(iter_to_str(_tgt)).encode("utf-8"))
					fwrtt.write(ens)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
