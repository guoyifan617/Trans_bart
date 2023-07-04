import torch
import sys
import tqdm
from utils.fmt.vocab.token import ldvocab


def bulid_zh_sort_vcb(zh_text,zh_vocab):
    with open(zh_text,'r',encoding='utf-8') as zh,open(zh_vocab,'w',encoding='utf-8') as zh_vcb:
        vcb_dict = {}
        for line in zh:
            tmp = line.strip()
            if tmp:
                for token in tmp:
                    vcb_dict[token] = vcb_dict.get(token, 0) + 1
        sort_dict = dict(sorted(vcb_dict.items(), key=lambda d: d[1], reverse=True))
        for k,v in sort_dict.items():
            zh_vcb.write(k)
            zh_vcb.write('\t')
            zh_vcb.write(str(v))
            zh_vcb.write('\n')

def load_zh_dict(file):
    vcb_zh = {}
    with open(file,'r',encoding='utf-8') as vcb:
        for line in vcb:
            tmp = line.strip()
            if tmp:
                tmp_list = tmp.split('\t')
                if len(tmp_list) == 2:
                    token = tmp_list[0]
                    value = tmp_list[1]
                    vcb_zh[token] = value
                else:
                    print('this list is :',tmp_list)
                    token = ' '
                    value = tmp_list[0]
                    vcb_zh[token] = value
    return vcb_zh

def load_bart_dict(file):
    vcb_bart = {}
    id = 0
    with open(file,'r',encoding='utf-8') as bart_file:
        for line in bart_file:
            token = line.strip()
            vcb_bart[token] = id
            id += 1
    return  vcb_bart


def replace_dict(zh_dict,mi_dict,zh_bart_dict):
    '''
    :param zh_dict: 使用大数据集做出来的排序后的中文字典，前5个是特殊token，用loadvcb读的
    :param mi_dict: 使用load_vcb得到的少数民族字典，前面5个是特殊token
    :param zh_bart_dict: 使用load_bart_dict得到的一个字典，这个字典得到的是中文bart的字典
    :return:
    '''
    zh_num = 5 #判定中文排序字典当前替换到哪一个
    mi_num = 5 #判定少数民族字典当前替换到哪一个
    tmp_zh_list = []
    tmp_mi_list = []
    replace_mi_vcb = {}
    #保存排序后的中文字典中的所有token
    for k in zh_dict.keys():
        tmp_zh_list.append(k)
    for _k in mi_dict.keys():
        tmp_mi_list.append(_k)
    print('zh_list 长度:',len(tmp_zh_list),'mi list长度:',len(tmp_mi_list))
    for _ in range(len(tmp_mi_list)-5):
        if tmp_zh_list[zh_num] in zh_bart_dict.keys():
            replace_mi_vcb[tmp_mi_list[mi_num]] = zh_bart_dict[tmp_zh_list[zh_num]]
            mi_num += 1
            zh_num += 1
        else:
            while tmp_zh_list[zh_num] not in zh_bart_dict.keys():
                zh_num += 1
            replace_mi_vcb[tmp_mi_list[mi_num]] = zh_bart_dict[tmp_zh_list[zh_num]]
            mi_num += 1
            zh_num += 1
    return replace_mi_vcb

def replace_dict_and_reverse_freq(zh_dict,mi_dict,zh_bart_dict):
    '''
    :param zh_dict: 使用大数据集做出来的排序后的中文字典，前5个是特殊token，用loadvcb读的
    :param mi_dict: 使用load_vcb得到的少数民族字典，前面5个是特殊token
    :param zh_bart_dict: 使用load_bart_dict得到的一个字典，这个字典得到的是中文bart的字典
    :return:
    '''
    zh_num = 5 #判定中文排序字典当前替换到哪一个
    mi_num = 0 #判定少数民族字典当前替换到哪一个
    tmp_zh_list = []
    tmp_mi_list = []
    replace_mi_vcb = {}
    #保存排序后的中文字典中的所有token
    sort_mi_dict = dict(sorted(mi_dict.items(), key=lambda d: d[1], reverse=True)) #频率从低到高！
    for k in zh_dict.keys():
        tmp_zh_list.append(k)
    for _k in sort_mi_dict.keys():
        tmp_mi_list.append(_k)
    print('zh_list 长度:',len(tmp_zh_list),'mi list长度:',len(tmp_mi_list))
    for _ in range(len(tmp_mi_list)-5):
        if tmp_zh_list[zh_num] in zh_bart_dict.keys():
            replace_mi_vcb[tmp_mi_list[mi_num]] = zh_bart_dict[tmp_zh_list[zh_num]]
            mi_num += 1
            zh_num += 1
        else:
            while tmp_zh_list[zh_num] not in zh_bart_dict.keys():
                zh_num += 1
            replace_mi_vcb[tmp_mi_list[mi_num]] = zh_bart_dict[tmp_zh_list[zh_num]]
            mi_num += 1
            zh_num += 1
    return replace_mi_vcb

if __name__ == '__main__':
    zh_dict,nwordz = ldvocab(sys.argv[1])
    mi_dict,nword = ldvocab(sys.argv[2])
    print(mi_dict)
    zh_bart_dict = load_bart_dict(sys.argv[3])
    replace_vcb = replace_dict_and_reverse_freq(zh_dict,mi_dict,zh_bart_dict)
    replace_vcb['<pad>'],replace_vcb['<sos>'],replace_vcb['<eos>'],replace_vcb['<unk>'],replace_vcb['<mask>'] = 0,101,102,100,103
    sort_replace_vcb = dict(sorted(replace_vcb.items(), key=lambda d: d[1], reverse=False))

    with open(sys.argv[4],'w',encoding='utf-8') as vcb:
        for k,v in sort_replace_vcb.items():
            vcb.write(k)
            vcb.write('\t')
            vcb.write(str(v))
            vcb.write('\n')

