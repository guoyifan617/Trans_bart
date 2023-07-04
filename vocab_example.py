from utils.fmt.vocab.token import ldvocab

vcb_16k_meng_bart = '/home/yfguo/CCMT_model/bart_wright/transformer-edge/cache/16k_Cache/meng2zh_bart/rs1/srcvcb.txt'
vcbi, nwordi_meng_bart_16k = ldvocab(vcb_16k_meng_bart)
print('bart 16k meng vcb size is :',nwordi_meng_bart_16k)

vcb_16k_wei_bart = '/home/yfguo/CCMT_model/bart_wright/transformer-edge/cache/16k_Cache/wei2zh_bart/rs1/srcvcb.txt'
vcbi, nwordi_wei_bart_16k = ldvocab(vcb_16k_wei_bart)
print('bart 16k wei vcb size is :',nwordi_wei_bart_16k)

vcb_16k_zang_bart = '/home/yfguo/CCMT_model/bart_wright/transformer-edge/cache/16k_Cache/zang2zh_bart/rs1/srcvcb.txt'
vcbi, nwordi_zang_bart_16k = ldvocab(vcb_16k_zang_bart)
print('bart 16k zang vcb size is :',nwordi_zang_bart_16k)

vcb_24_meng_bart = '/home/yfguo/CCMT_model/bart_wright/transformer-edge/cache/24k_cache/meng2zh_bart/rs1/srcvcb.txt'
vcbi, nwordi_meng_bart_24k = ldvocab(vcb_24_meng_bart)
print('bart 24k meng vcb size is :',nwordi_meng_bart_24k)

vcb_24k_wei_bart = '/home/yfguo/CCMT_model/bart_wright/transformer-edge/cache/24k_cache/wei2zh_bart/rs1/srcvcb.txt'
vcbi, nwordi_wei_bart_24k = ldvocab(vcb_24k_wei_bart)
print('bart 24k wei vcb size is :',nwordi_wei_bart_24k)

vcb_24k_zang_bart = '/home/yfguo/CCMT_model/bart_wright/transformer-edge/cache/24k_cache/zang2zh_bart/rs1/srcvcb.txt'
vcbi, nwordi_zang_bart_24k = ldvocab(vcb_24k_zang_bart)
print('bart 24k zang vcb size is :',nwordi_zang_bart_24k)