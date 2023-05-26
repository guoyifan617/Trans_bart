#encoding: utf-8

from torch.nn import ModuleList

from modules.base import Dropout, PositionwiseFF as PositionwiseFFBase
from modules.nas.ffn import PositionwiseFF
from modules.nas.gdart import Cell
from utils.base import add_module
from utils.train.base import unfreeze_module

def share_cell(netin, share_all=False, share_global=True):

	_base_cell = None
	_reset_base_cell = not share_global
	for net in netin.modules():
		if isinstance(net, ModuleList):
			if _reset_base_cell:
				_base_cell = None
			for _module in net.modules():
				if isinstance(_module, PositionwiseFF):
					if _base_cell is None:
						_base_cell = _module.net
					else:
						if share_all:
							_module.net = _base_cell
						else:
							_cell = _module.net
							for _cnode, _bnode in zip(_cell.nets, _base_cell.nets):
								_cnode.weight = _bnode.weight

	return netin

def is_nas(netin):

	rs = False
	for net in netin.modules():
		if isinstance(net, Cell):
			rs = True
			break

	return rs

def unfreeze_cell(netin):

	for net in netin.modules():
		if isinstance(net, Cell):
			unfreeze_module(net)

	return netin

def get_ffn_attr(ffn):

	_hsize, _isize = ffn.net[0].weight.size()
	_drop = _act_drop = 0.0
	_ = list(ffn.modules())
	_lind = len(_) - 1
	for i, _m in enumerate(_):
		if isinstance(_m, Dropout):
			if i < _lind:
				_act_drop = _m.p
			else:
				_drop = _m.p

	return _isize, _hsize, _drop, _act_drop, ffn.norm_residual

def patch_stdffn(netin):

	for _name, _module in netin.named_modules():
		if isinstance(_module, PositionwiseFFBase):
			_isize, _hsize, _drop, _act_drop, _nr = get_ffn_attr(_module)
			add_module(netin, _name, PositionwiseFF(_isize, hsize=_hsize, dropout=_drop, act_drop=_act_drop, norm_residual=_nr))

	return netin
