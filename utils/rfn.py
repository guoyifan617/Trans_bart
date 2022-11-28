#encoding: utf-8

from torch.nn import ModuleList

from modules.rfn import LSTMCell4FFN
from utils.base import add_module

def share_LSTMCell(netin, share_all=False):

	for net in netin.modules():
		if isinstance(net, ModuleList):
			_base_net = None
			for _name, _module in net.named_modules():
				if isinstance(_module, LSTMCell4FFN):
					if _base_net is None:
						_base_net = _module
					else:
						if share_all:
							add_module(net, _name, _base_net)
						else:
							_module.trans, _module.normer = _base_net.trans, _base_net.normer

	return netin
