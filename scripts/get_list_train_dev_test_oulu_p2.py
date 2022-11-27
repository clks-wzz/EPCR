import os
import glob

from ClassOulu import IJCB

dicts_path = {
    'train': {
            'data': '/share/wangzezheng/data/liveness/public/OULU/IJCB/Train_images',
            'save': '/share/wangzezheng/data/liveness/public/OULU/IJCB/txts/final/list_semi_train_oulu_p2.txt'
        },
    'dev': {
            'data': '/share/wangzezheng/data/liveness/public/OULU/IJCB/Dev_images',
            'save': '/share/wangzezheng/data/liveness/public/OULU/IJCB/txts/final/list_semi_dev_oulu_p2.txt'
        },
    'test': {
            'data': '/share/wangzezheng/data/liveness/public/OULU/IJCB/Test_images',
            'save': '/share/wangzezheng/data/liveness/public/OULU/IJCB/txts/final/list_semi_test_oulu_p2.txt'
        },
    'dev_test': {
            'data': None,
            'save': '/share/wangzezheng/data/liveness/public/OULU/IJCB/txts/final/list_semi_dev_test_oulu_p2.txt'
        }
}

protocol = 'ijcb_protocal_2'

def main():
    dataObj_train = IJCB(protocol=protocol, mode='train')
    dataObj_train.process_train(dicts_path['train']['data'], dicts_path['train']['save'])

    dataObj_dev= IJCB(protocol=protocol, mode='dev')
    lines_dev = dataObj_dev.process_dev(dicts_path['dev']['data'], None, only_return_lines=True)

    dataObj_test= IJCB(protocol=protocol, mode='test')
    lines_test = dataObj_test.process_test(dicts_path['test']['data'], None, only_return_lines=True)

    lines_dev_test = lines_dev + lines_test
    print(len(lines_dev), len(lines_test), len(lines_dev_test))
    with open(dicts_path['dev_test']['save'], 'w') as fid:
        fid.writelines(lines_dev_test)

if __name__ == '__main__':
    main()