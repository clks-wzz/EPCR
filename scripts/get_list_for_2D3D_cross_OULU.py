import os
import glob 
import numpy as np 
from tqdm import tqdm

#path_src = '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_train_OCM-I.txt'
#path_dst = '/share/wangzezheng/data/liveness/public/OULU/IJCB/txts/final/list_2D_to_3D_OUlU_train.txt'
path_src = '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_test_CIM-O.txt'
path_dst = '/share/wangzezheng/data/liveness/public/OULU/IJCB/txts/final/list_2D_to_3D_OUlU_test.txt'

def get_loc_str(loc_path):
    with open(loc_path, 'r') as fid:
        lines = fid.readlines()
    
    x1, y1, w, h = [float(x.strip()) for x in lines[:4]]
    x2 = x1 + w
    y2 = y1 + h
    loc_str = ' '.join([str(x) for x in [x1, y1, x2, y2]])
    return loc_str

def main_master():
    with open(path_src, 'r') as fid:
        lines = fid.readlines()
    
    lines_res = []
    for l, line in enumerate(tqdm(lines)):
        name_splits = line.strip().split()
        img_path = name_splits[0]
        loc_path = name_splits[1]
        label_str = name_splits[2]
        
        if 'OULU/IJCB' not in img_path:
            continue

        loc_str = get_loc_str(loc_path)

        if label_str == '0':
            label = 1
        else:
            label = 0

        line_new = ' '.join([img_path, loc_str, str(label)]) + '\n'

        lines_res.append(line_new)
    
    with open(path_dst, 'w') as fid:
        fid.writelines(lines_res)

def main_test():
    fix_num = 8

    with open(path_dst, 'r') as fid:
        lines = fid.readlines()

    dir2lines = {}
    for l, line in enumerate(tqdm(lines)):
        img_path = line.strip().split()[0]
        dir_name = img_path.split('/')[-2]
        dir2lines.setdefault(dir_name, [])
        dir2lines[dir_name].append(line)
    
    lines_res = []
    for dir_name in dir2lines.keys():
        lines = dir2lines[dir_name]
        sample_div_len = len(lines) // fix_num
        lines_res += lines[::sample_div_len]
    
    with open(path_dst + '.eco', 'w') as fid:
        fid.writelines(lines_res)

if __name__ == '__main__':
    main_test()