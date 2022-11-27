import os 
import glob 

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
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_train_extra_CIM-O.txt',
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_train_extra_OIM-C.txt',
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_train_extra_OCM-I.txt',
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_train_extra_OCI-M.txt',
]
list_dst_test = [
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_test_CIM-O.txt',
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_test_OIM-C.txt',
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_test_OCM-I.txt',
    '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_test_OCI-M.txt',
]

def main():
    for i in range(len(list_train)):
        dst_train = list_dst_train[i]
        dst_test = list_dst_test[i]

        lines_train = []
        lines_test = []
        for a in range(len(list_train)):
            if a==i:
                path_train = list_train[a]
                with open(path_train, 'r') as fid:
                    lines = fid.readlines()
                
                lines_processed = [' '.join(e.strip().split()[:3] + ['0']) + '\n' for e in lines]
                print('extra:', path_train, len(lines_processed))
                lines_train += lines_processed
            else:
                path_train = list_train[a]
                with open(path_train, 'r') as fid:
                    lines = fid.readlines()
                print('intra:', path_train, len(lines))
                lines_train += lines
        
        for b in range(len(list_test)):
            if b==i:
                path_test = list_test[b]
                with open(path_test, 'r') as fid:
                    lines = fid.readlines()
                print(path_test, len(lines))
                lines_test += lines
        
        with open(dst_train, 'w') as fid:
            fid.writelines(lines_train)

        '''    
        with open(dst_test, 'w') as fid:
            fid.writelines(lines_test)
        '''

        print('Out:')
        print(dst_train)
        print('Train:', len(lines_train))
        print(dst_test)
        print('Test:', len(lines_test))
        




if __name__ == '__main__':
    main()