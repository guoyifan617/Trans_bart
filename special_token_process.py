#encoding: utf-8
import sys

def handles(src,tgt):
    with open(src,'r',encoding='utf-8') as src_file, open(tgt,'w',encoding='utf-8') as tgt_file:
        for line in src_file:
            tmp = line.strip()
            if tmp:
                for token in tmp.split():
                    if token =='[SEP]' or token =='[CLS]':
                        continue
                    else:
                        tgt_file.write(token)
                tgt_file.write('\n')

handles(sys.argv[1],sys.argv[2])