#encoding: utf-8

from tqdm import tqdm

from utils.fmt.plm.custbert.raw.ploader import Loader as DataLoader

tdl = DataLoader(["train.py", "LICENSE", "transformer/Decoder.py"], ["train.py", "LICENSE", "translator.py", "transformer/Decoder.py"], "cache/test/common.vcb", num_cache=32, raw_cache_size=2048, nbatch=128, print_func=None)

for _ in tqdm(tdl()):
	pass
