#encoding: utf-8

import torch

from utils.tqdm import tqdm

bsize = 512
beam_size = 1280
vsize = beam_size + 10
niter = 10000

real_bsize = bsize * beam_size
beam_size2 = beam_size * beam_size
bsizeb2 = bsize * beam_size2

out = torch.randn(bsize * beam_size, 1, vsize)
_scores, _wds = out.topk(beam_size, dim=-1)
scores, _inds = _scores.view(bsize, beam_size2).topk(beam_size, dim=-1)

for i in tqdm(range(niter)):
	wds_gather = _wds.view(bsize, beam_size2).gather(-1, _inds).view(real_bsize, 1)

for i in tqdm(range(niter)):
	_tinds = (_inds + torch.arange(0, bsizeb2, beam_size2, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)
	wds = _wds.view(bsizeb2).index_select(0, _tinds).view(real_bsize, 1)
