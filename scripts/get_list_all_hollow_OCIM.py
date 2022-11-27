import os 
import glob 

marks = ['O', 'C', 'I', 'M']

list_train = [
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_train_oulu.txt',
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_train_casia.txt',
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_train_replayattack.txt',
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_train_msu.txt',
]
list_test = [
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_test_oulu.txt',
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_test_casia.txt',
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_test_replayattack.txt',
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_test_msu.txt',
]
list_dst_train = [
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_train_CIM-O.txt.hollow.',
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_train_OIM-C.txt.hollow.',
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_train_OCM-I.txt.hollow.',
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_train_OCI-M.txt.hollow.',
]
list_dst_test = [
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_test_CIM-O.txt',
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_test_OIM-C.txt',
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_test_OCM-I.txt',
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_test_OCI-M.txt',
]

def convert_to_hollow(lines):
    lines_res = []
    for l, line in enumerate(lines):
        name_splits = line.strip().split()
        name_splits[-1] = '0'
        line_new = ' '.join(name_splits) + '\n'
        lines_res.append(line_new)
    
    return lines_res

def main():
    for i in range(len(list_train)):
        dst_train = list_dst_train[i]
        dst_test = list_dst_test[i]

        
        
        for t in range(len(marks)):
            if t==i:
                continue
                
            lines_train = []
            lines_test = []
            save_path = dst_train + marks[t]
            for a in range(len(list_train)):
                if a==i:
                    continue
                elif a==t:
                    path_train = list_train[a]
                    with open(path_train, 'r') as fid:
                        lines = fid.readlines()
                    lines = convert_to_hollow(lines)
                    print(path_train, len(lines))
                    lines_train += lines
                else:
                    path_train = list_train[a]
                    with open(path_train, 'r') as fid:
                        lines = fid.readlines()
                    print(path_train, len(lines))
                    lines_train += lines
        
            with open(save_path, 'w') as fid:
                fid.writelines(lines_train)

            print('Out:')
            print(save_path)
            print('Train:', len(lines_train))
        




if __name__ == '__main__':
    main()