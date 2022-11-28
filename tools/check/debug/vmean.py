#encoding: utf-8

import sys

from utils.h5serial import h5load

rsm = h5load(sys.argv[1])

for k, v in rsm.items():
	if k.find("wemb") > 0:
		print(v.mean())
