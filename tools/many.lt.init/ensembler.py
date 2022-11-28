#encoding: utf-8

""" usage:
	python tools/many.lt.init/ensembler.py $perf.txt $expm_path $rsm.h5
"""

import sys
from math import ceil

from utils.fmt.base import dict_insert_list, iter_dict_sort, load_objects
from utils.h5serial import h5load, h5save
from utils.torch.comp import secure_type_map

from cnfg.ihyp import *

def get_topkf(perfd, k):

	data = {}
	# v: (terr, vloss, vprec, init_vloss, init_vprec,)
	for i, (terr, vloss, vprec, init_vloss, init_vprec,) in perfd.items():
		data = dict_insert_list(data, "model_%d_%.3f_%.3f_%.2f.h5" % (i, terr, vloss, vprec,), vprec, vloss, terr)

	rs = []
	_nremain = k
	for tmp in iter_dict_sort(data, free=True):
		_nd = len(tmp)
		if _nd <= _nremain:
			rs.extend(tmp)
		else:
			_nd = _nremain
			rs.extend(tmp[:_nd])
		_nremain -= _nd
		if _nremain <= 0:
			break

	return rs

def prune_parameter(para, ratio, inplace=True):

	para_abs = para.abs()
	_mv = para_abs.view(-1).sort(descending=True)[0][ceil(para.numel() * ratio) - 1].item()
	_m = para_abs.lt(_mv)

	return para.masked_fill_(_m, 0.0) if inplace else para.masked_fill(_m, 0.0)

def handle(perf, wkd, rsf, k=10, prune_ratio=None, avg=True):

	srcfl = get_topkf(load_objects(perf), k)

	rsm = h5load(wkd + srcfl[0])

	src_type = [para.dtype for para in rsm]
	map_type = [secure_type_map[para.dtype] if para.dtype in secure_type_map else None for para in rsm]
	sec_rsm = [para if typ is None else para.to(typ, non_blocking=True) for para, typ in zip(rsm, map_type)]
	if prune_ratio is not None:
		sec_rsm = [prune_parameter(para, prune_ratio, inplace=True) for para in sec_rsm]
	nmodel = 1
	for modelf in srcfl[1:]:
		for basep, mpload, typ in zip(sec_rsm, h5load(wkd + modelf), map_type):
			_para = mpload if typ is None else mpload.to(typ, non_blocking=True)
			if prune_ratio is not None:
				_para = prune_parameter(_para, prune_ratio, inplace=True)
			basep.add_(_para)
		nmodel += 1

	if avg and (nmodel > 1):
		nmodel = float(nmodel)
		for basep in sec_rsm:
			basep.div_(nmodel)

	rsm = [para if mtyp is None else para.to(styp, non_blocking=True) for para, mtyp, styp in zip(sec_rsm, map_type, src_type)]

	h5save(rsm, rsf, h5args=h5zipargs)

if __name__ == "__main__":
	_nargs = len(sys.argv)
	if _nargs == 4:
		handle(sys.argv[1], sys.argv[2], sys.argv[3])
	elif _nargs == 5:
		handle(sys.argv[1], sys.argv[2], sys.argv[3], k=int(sys.argv[4]))
	else:
		handle(sys.argv[1], sys.argv[2], sys.argv[3], k=int(sys.argv[4]), prune_ratio=float(sys.argv[5]))
