#encoding: utf-8

import sys
from numpy import array as np_array, int32 as np_int32, uint8 as np_uint8

from utils.fmt.base import sys_open
from utils.h5serial import h5File

def parse_bpe_list(lin):

	rs = []
	for lu in lin:
		tmp = lu.split("\t")
		rs.append([tmpu.split() for tmpu in tmp])

	return rs

# lbpe: (bsize, [[tok, ...], ...])
def build_ind_mask(lbpe, seql):

	mxtk = mxp = 0
	for seq in lbpe:
		l = len(seq)
		if l > mxp:
			mxp = l
		for phrase in seq:
			l = len(phrase)
			if l > mxtk:
				mxtk = l

	# mxp * mxtk
	ind = []
	mask = []
	cur_b = 0
	for seq in lbpe:
		curid = curb * seql + 1
		_mask = []
		for phrase in seq:
			lp = len(phrase)
			if lp < mxtk:
				ind.extend(list(range(curid:curid+lp)) + [0 for i in range(mxtk - lp)])
				_mask.append([0 for i in range(lp)] + [1 for i in range(mxtk - lp)])
			else:
				ind.extend(list(range(curid:curid+lp)))
				_mask.append([0 for i in range(lp)])
			curid += lp
		_ls = len(seq)
		if _ls < mxp:
			ind.extend([0 for i in range(mxtk * (mxp - _ls))])
			_mask.extend([[1 for i in range(mxtk)] for j in range(mxp - _ls)])
		mask.append(_mask)
		curb += 1

	# ind: (bsize * mxp * mxtk)
	# mask: (bsize, mxp, mxtk)
	return ind, mask

def handle(h5src, bpeparsf, rsf):

	with h5File(h5src, "r") as data, h5File(rsf, "w", libver=h5_libver) as frs, sys_open(bpeparsf, "rb") as frd:
		ndata = data["ndata"][()].item()
		src_grp, ind_grp, mask_grp = data["src"], frs.create_group("ind"), frs.create_group("mask4phrase")
		for i in range(ndata):
			sid = str(i)
			bsize, seql = src_grp[sid].shape
			bped = parse_bpe_list([frd.readline().strip().decode("utf-8") for i in range(bsize)])
			ind, mask = build_ind_mask(bped, seql)
			ind_grp[sid] = np_array(ind, dtype=np_int32)
			mask_grp[sid] = np_array(mask, dtype=np_uint8)
		frs["ndata"] = np_array([ndata], dtype=np_int32)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3])
