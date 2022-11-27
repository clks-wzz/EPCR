import os
import glob

from ClassOulu import IJCB

dicts_path = {
    'train': {
            'data': '/share/wangzezheng/data/liveness/public/OULU/IJCB/Train_images',
            'save': '/share/wangzezheng/data/liveness/public/OULU/IJCB/txts/final/list_semi_train_oulu_p3.txt'
        },
    'dev': {
            'data': '/share/wangzezheng/data/liveness/public/OULU/IJCB/Dev_images',
            'save': '/share/wangzezheng/data/liveness/public/OULU/IJCB/txts/final/list_semi_dev_oulu_p3.txt'
        },
    'test': {
            'data': '/share/wangzezheng/data/liveness/public/OULU/IJCB/Test_images',
            'save': '/share/wangzezheng/data/liveness/public/OULU/IJCB/txts/final/list_semi_test_oulu_p3.txt'
        },
    'dev_test': {
            'data': None,
            'save': '/share/wangzezheng/data/liveness/public/OULU/IJCB/txts/final/list_semi_dev_test_oulu_p3.txt'
        }
}

protocol = 'ijcb_protocal_3'

def main():
    for i in range(6):
        p = i+1
        _protocol = protocol + '_%d'%(p)

        dataObj_train = IJCB(protocol=_protocol, mode='train')
        dataObj_train.process_train(dicts_path['train']['data'], dicts_path['train']['save'][:-4] + '_%d.txt'%(p))

        dataObj_dev= IJCB(protocol=_protocol, mode='dev')
        lines_dev = dataObj_dev.process_dev(dicts_path['dev']['data'], None, only_return_lines=True)

        dataObj_test= IJCB(protocol=_protocol, mode='test')
        lines_test = dataObj_test.process_test(dicts_path['test']['data'], None, only_return_lines=True)

        lines_dev_test = lines_dev + lines_test
        print(len(lines_dev), len(lines_test), len(lines_dev_test))
        with open(dicts_path['dev_test']['save'][:-4] + '_%d.txt'%(p), 'w') as fid:
            fid.writelines(lines_dev_test)

if __name__ == '__main__':
    main()