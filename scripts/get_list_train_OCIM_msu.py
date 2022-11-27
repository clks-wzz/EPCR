import os
import glob

path_imgs = '/share/wangzezheng/data/liveness/public/MSU/MSU-MFSD/MSU-MFSD/MSU_imgs'
path_dst  = '/share/wangzezheng/data/liveness/public/OCIM/txts/final/list_semi_ocim_train_msu.txt'
path_limit = '/share/wangzezheng/data/liveness/public/MSU/MSU-MFSD/MSU-MFSD/train_sub_list.txt'

frame_interval = 4

def main():
    with open(path_limit, 'r') as fid:
        lines = fid.readlines()
    limits = []
    for line in lines:
        limits.append(float(line.strip()))

    FILES_0 = glob.glob(os.path.join(path_imgs, '*'))
    FILES = []
    for file_path in FILES_0:
        FILES += glob.glob(os.path.join(file_path, '*'))

    lines_res = []

    _pass = 0

    for f, file_path in enumerate(FILES):
        file_name = os.path.split(file_path)[-1]
        client = file_name.split('_')[1][len('client'):]
        client = float(client)
        if client in limits:
            _pass += 1.
            pass
        else:
            #print('Continue:', file_path)
            continue

        PAI = file_name.split('_')[0]
        if PAI == 'real':
            true_label = '1'
        else:
            true_label = '0'

        DATS = glob.glob(os.path.join(file_path, '*_scene.dat'))
        DATS = sorted(DATS, key=lambda x: float(x.split('/')[-1].split('_')[-2]))

        for i in range(0, len(DATS), frame_interval):
            dat_path = DATS[i]
            img_path = dat_path[:-4] + '.jpg'
            if not os.path.exists(img_path):
                continue
            line_new = img_path + ' ' + dat_path + ' ' + true_label + ' 1\n' 
            lines_res.append(line_new)
    
    print(_pass, len(FILES))

    with open(path_dst, 'w') as fid:
        fid.writelines(lines_res)
    
    print('Out:', path_dst)

if __name__ == '__main__':
    main()