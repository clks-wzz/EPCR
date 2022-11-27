import os
import glob

#path_imgs = '/share/wangzezheng/data/liveness/public/OULU/IJCB/Train_images'
#path_dst  = '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_train_oulu.txt'
path_imgs = '/share/wangzezheng/data/liveness/public/OULU/IJCB/Test_images'
path_dst  = '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_test_oulu.txt'

frame_interval = 4

def main():
    FILES = glob.glob(os.path.join(path_imgs, '*'))

    lines_res = []

    for f, file_path in enumerate(FILES):
        if 'ReplayAttack' in file_path or 'CASIA' in file_path:
            continue
        file_name = os.path.split(file_path)[-1]
        PAI = file_name.split('_')[-1]
        if PAI == '1':
            true_label = '1'
        else:
            true_label = '0'

        #IMGS = glob.glob(os.path.join(file_path, '*_scene.jpg'))
        #IMGS = sorted(IMGS, key=lambda x: float(x.split('/')[-1].split('_')[-2]))
        DATS = glob.glob(os.path.join(file_path, '*_scene.dat'))
        DATS = sorted(DATS, key=lambda x: float(x.split('/')[-1].split('_')[-2]))

        for i in range(0, len(DATS), frame_interval):
            dat_path = DATS[i]
            img_path = dat_path[:-4] + '.jpg'
            line_new = img_path + ' ' + dat_path + ' ' + true_label + ' 1\n' 
            lines_res.append(line_new)
    
    with open(path_dst, 'w') as fid:
        fid.writelines(lines_res)
    
    print('Out:', path_dst)

if __name__ == '__main__':
    main()