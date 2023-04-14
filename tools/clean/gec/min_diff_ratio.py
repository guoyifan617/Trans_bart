#encoding: utf-8

import sys

from utils.fmt.base import sys_open
from utils.fmt.diff import seq_diff_ratio

def handle(srcfs, srcft, tgtfs, tgtft, ratio, splt):

	ens = "\n".encode("utf-8")
	with sys_open(srcfs, "rb") as frs, sys_open(srcft, "rb") as frt, sys_open(tgtfs, "wb") as fws, sys_open(tgtft, "wb") as fwt:
		for sline, tline in zip(frs, frt):
			sline, tline = sline.strip(), tline.strip()
			if sline and tline:
				sline, tline = sline.decode("utf-8"), tline.decode("utf-8")
				if seq_diff_ratio(*((sline.split(), tline.split(),) if splt else (sline, tline,))) >= ratio:
					fws.write(sline.encode("utf-8"))
					fws.write(ens)
					fwt.write(tline.encode("utf-8"))
					fwt.write(ens)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), int(sys.argv[6]))
