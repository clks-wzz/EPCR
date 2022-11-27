import os
import glob

# img[loc[1]:loc[1] + loc[3], loc[0]: loc[0] + loc[2],:]

path_data = '/share/wangzezheng/data/liveness/public/OULU/IJCB/Test_images'
path_txt = '/share/wangzezheng/data/liveness/public/OULU/IJCB/txts/final/list_semi_test_oulu_p1.txt'

limit_sessions = [3]
frame_num = 8
#frame_interval = 4

def main():
    FILES = glob.glob(os.path.join(path_data, '*'))
    print(len(FILES))
    lines_all = []
    for f, file_path in enumerate(FILES):
        file_name = os.path.split(file_path)[-1]
        #print(file_path)
        try:
            id_phone, id_session, id_user, id_pai = file_name.split('_')
        except:
            print('err file name:', file_name)
            continue
        session = int(float(id_session))
        if session in limit_sessions:
            supervised = 1
        else:
            continue
        TXTS = glob.glob(os.path.join(file_path, '*_scene.dat'))
        TXTS = sorted(TXTS, key=lambda x: float(x.split('/')[-1].split('_')[4]))
        TXTS_new = []

        frame_interval = int(len(TXTS)) // frame_num

        for i in range(0, len(TXTS), frame_interval):
            TXTS_new.append(TXTS[i])

        lines_file = []
        for t, txt_path in enumerate(TXTS_new):
            img_path = txt_path[:-4] + '.jpg'
            line_new = img_path + ' ' + txt_path + ' ' + file_name + ' ' + 'test' + ' '  +  '\n'
            lines_file.append(line_new)

        print(f, len(FILES), file_name, len(lines_file))

        lines_all += lines_file
    
    with open(path_txt, 'w') as fid:
        fid.writelines(lines_all)

if __name__ == '__main__':
    main()