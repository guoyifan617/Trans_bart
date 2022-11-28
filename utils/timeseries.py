#encoding: utf-8

from torch import Tensor

def repeat_bsize_for_beam_tensor(tin, beam_size, ngroup):

	_tsize = list(tin.size())
	_rarg = [1 for i in range(len(_tsize) + 1)]
	_rarg[1] = beam_size
	_ = _tsize[0] // ngroup
	_tsize[0] *= beam_size

	return tin.view(_, ngroup, *_tsize[1:]).repeat(*_rarg).view(_tsize)

def expand_bsize_for_beam(*inputs, beam_size=1, ngroup=None):

	outputs = []
	for inputu in inputs:
		if isinstance(inputu, Tensor):
			outputs.append(repeat_bsize_for_beam_tensor(inputu, beam_size, ngroup))
		elif isinstance(inputu, dict):
			outputs.append({k: expand_bsize_for_beam(v, beam_size=beam_size, ngroup=ngroup) for k, v in inputu.items()})
		elif isinstance(inputu, tuple):
			outputs.append(tuple(expand_bsize_for_beam(tmpu, beam_size=beam_size, ngroup=ngroup) for tmpu in inputu))
		elif isinstance(inputu, list):
			outputs.append([expand_bsize_for_beam(tmpu, beam_size=beam_size, ngroup=ngroup) for tmpu in inputu])
		else:
			outputs.append(inputu)

	return outputs[0] if len(inputs) == 1 else tuple(outputs)

def index_tensors(*inputs, indices=None, dim=0, ngroup=None):

	outputs = []
	for inputu in inputs:
		if isinstance(inputu, Tensor):
			_size = list(inputu.size())
			_ = _size[1:]
			outputs.append(inputu.view(_size[0] // ngroup, ngroup, *_).index_select(dim, indices).view(-1, *_))
		elif isinstance(inputu, dict):
			outputs.append({k: index_tensors(v, indices=indices, dim=dim, ngroup=ngroup) for k, v in inputu.items()})
		elif isinstance(inputu, tuple):
			outputs.append(tuple(index_tensors(tmpu, indices=indices, dim=dim, ngroup=ngroup) for tmpu in inputu))
		elif isinstance(inputu, list):
			outputs.append([index_tensors(tmpu, indices=indices, dim=dim, ngroup=ngroup) for tmpu in inputu])
		else:
			outputs.append(inputu)

	return outputs[0] if len(inputs) == 1 else tuple(outputs)
