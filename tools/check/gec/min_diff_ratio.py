#encoding: utf-8

import sys

from utils.fmt.base import sys_open
from utils.fmt.diff import seq_diff_ratio

def handle(srcfs, srcft, splt=1, dbg=False):

	min_v = 1.0
	with sys_open(srcfs, "rb") as fs, sys_open(srcft, "rb") as ft:
		for sline, tline in zip(fs, ft):
			sline, tline = sline.strip(), tline.strip()
			if sline and tline:
				sline, tline = sline.decode("utf-8"), tline.decode("utf-8")
				if splt:
					sline, tline = sline.split(), tline.split()
				_ratio = seq_diff_ratio(sline, tline)
				if _ratio < min_v:
					min_v = _ratio
					if dbg:
						print(_ratio)
						print(sline)
						print(tline)
					if _ratio == 0.0:
						break

	return min_v

if __name__ == "__main__":
	print(handle(sys.argv[1], sys.argv[2], int(sys.argv[3])))
