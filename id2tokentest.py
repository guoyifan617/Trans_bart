from utils.fmt.vocab.token import ldvocab

fvocab_i = '/home/yfguo/CCMT_model/bart_wright/transformer-edge/cache/16k_Cache/meng2zh_bart/rs1/srcvcb.txt'
vcbi, nwordi = ldvocab(fvocab_i)
with open('ch_bart_vocab.txt','w',encoding='utf-8') as vcb:
    for k,v in vcbi.items():
        vcb.write(k)
        vcb.write('\t')
        vcb.write(str(v))
        vcb.write('\n')
