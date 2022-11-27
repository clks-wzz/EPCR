# lucky
import os
import glob
import copy

class IJCB:
    def __init__(self, protocol, mode):    

        protocol_dict = {}
        protocol_dict['ijcb_protocal_1']={'train': { 'session': [1, 2], 'phones': [1, 2, 3, 4, 5, 6], 
                                                    'users': list(range(1,21)), 'PAI': [1, 2, 3, 4, 5] },
                                          'dev': { 'session': [1, 2], 'phones': [1, 2, 3, 4, 5, 6], 
                                                    'users': list(range(21,36)), 'PAI': [1, 2, 3, 4, 5] },
                                          'test': { 'session': [3], 'phones': [1, 2, 3, 4, 5, 6], 
                                                    'users': list(range(36,56)), 'PAI': [1, 2, 3, 4, 5] }
                                        }
        protocol_dict['ijcb_protocal_2']={'train': { 'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5, 6], 
                                                    'users': list(range(1,21)), 'PAI': [1, 2, 4] },
                                          'dev': { 'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5, 6], 
                                                    'users': list(range(21,36)), 'PAI': [1, 2, 4] },
                                          'test': { 'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5, 6], 
                                                    'users': list(range(36,56)), 'PAI': [1, 3, 5] }
                                        }        
        protocol_dict['ijcb_protocal_3']={'train': { 'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5], 
                                                    'users': list(range(1,21)), 'PAI': [1, 2, 3, 4, 5] },
                                          'dev': { 'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5], 
                                                    'users': list(range(21,36)), 'PAI': [1, 2, 3, 4, 5] },
                                          'test': { 'session': [1, 2, 3], 'phones': [6], 
                                                    'users': list(range(36,56)), 'PAI': [1, 2, 3, 4, 5] }
                                        }
        for i in range(6):
            protocol_dict['ijcb_protocal_3_%d'%(i+1)] = copy.deepcopy(protocol_dict['ijcb_protocal_3'])
            protocol_dict['ijcb_protocal_3_%d'%(i+1)]['train']['phones'] = []
            protocol_dict['ijcb_protocal_3_%d'%(i+1)]['dev']['phones'] = []
            protocol_dict['ijcb_protocal_3_%d'%(i+1)]['test']['phones'] = []
            for j in range(6):
                if j==i:
                    protocol_dict['ijcb_protocal_3_%d'%(i+1)]['test']['phones'].append(j+1)
                else:
                    protocol_dict['ijcb_protocal_3_%d'%(i+1)]['train']['phones'].append(j+1)
                    protocol_dict['ijcb_protocal_3_%d'%(i+1)]['dev']['phones'].append(j+1)

        protocol_dict['ijcb_protocal_4']={'train': { 'session': [1, 2], 'phones': [1, 2, 3, 4, 5], 
                                                    'users': list(range(1,21)), 'PAI': [1, 2, 4] },
                                          'dev': { 'session': [1, 2], 'phones': [1, 2, 3, 4, 5], 
                                                    'users': list(range(21,36)), 'PAI': [1, 2, 4] },
                                          'test': { 'session': [3], 'phones': [6], 
                                                    'users': list(range(36,56)), 'PAI': [1, 3, 5] }
                                        }
        for i in range(6):
            protocol_dict['ijcb_protocal_4_%d'%(i+1)] = copy.deepcopy(protocol_dict['ijcb_protocal_4'])
            protocol_dict['ijcb_protocal_4_%d'%(i+1)]['train']['phones'] = []
            protocol_dict['ijcb_protocal_4_%d'%(i+1)]['dev']['phones'] = []
            protocol_dict['ijcb_protocal_4_%d'%(i+1)]['test']['phones'] = []
            for j in range(6):
                if j==i:
                    protocol_dict['ijcb_protocal_4_%d'%(i+1)]['test']['phones'].append(j+1)
                else:
                    protocol_dict['ijcb_protocal_4_%d'%(i+1)]['train']['phones'].append(j+1)
                    protocol_dict['ijcb_protocal_4_%d'%(i+1)]['dev']['phones'].append(j+1)
        
        protocol_dict['ijcb_protocal_all']={'train': { 'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5, 6], 
                                                    'users': list(range(1,21)), 'PAI': [1, 2, 3, 4, 5] },
                                          'dev': { 'session': [1, 2, 3], 'phones': [1, 2, 3, 4, 5, 6], 
                                                    'users': list(range(21,36)), 'PAI': [1, 2, 3, 4, 5] },
                                          'test': { 'session': [3], 'phones': [1, 2, 3, 4, 5, 6], 
                                                    'users': list(range(36,56)), 'PAI': [1, 2, 3, 4, 5] }
                                        }

        self.protocol_dict = protocol_dict
        self.mode = mode      

        if not (protocol in self.protocol_dict.keys()):
            print('error: Protocal should be ', list(self.protocol_dict.keys()) )
            exit(1)
        self.protocol = protocol
        self.protocol_info = protocol_dict[protocol][mode]

    def isInPotocol(self, file_name_full):
        file_name = os.path.split(file_name_full)[-1]
        name_split = file_name.split('_')
        if not len(name_split)==4:
            return False
        
        [phones_, session_, users_, PAI_] = [int(x) for x in name_split]

        if (phones_ in self.protocol_info['phones']) and (session_ in self.protocol_info['session']) \
                and (users_ in self.protocol_info['users']) and (PAI_ in self.protocol_info['PAI']):
            return True
        else:
            return False

    def dataset_process(self, file_list):
        res_list = []
        for i in range(len(file_list)):
            file_name_full = file_list[i]
            if self.isInPotocol(file_name_full):
                res_list.append(file_name_full)
        print('Dataset Info:')
        print('----------------------------------------')
        print('IJCB', self.protocol, self.mode)
        print('File Counts:', len(res_list))
        print('----------------------------------------')

        return res_list
    
    def process_train(self, path_data, path_txt, frame_interval=4):
        FILES = glob.glob(os.path.join(path_data, '*'))
        print(len(FILES))
        lines_all = []
        for f, file_path in enumerate(FILES):
            file_name = os.path.split(file_path)[-1]
            #id_phone, id_session, id_user, id_pai = file_name.split('_')
            #session = int(float(id_session))
            #if session in limit_sessions:
            if self.isInPotocol(file_path):
                supervised = 1
            else:
                supervised = 0
            TXTS = glob.glob(os.path.join(file_path, '*_scene.dat'))
            TXTS = sorted(TXTS, key=lambda x: float(x.split('/')[-1].split('_')[4]))
            TXTS_new = []
            for i in range(0, len(TXTS), frame_interval):
                TXTS_new.append(TXTS[i])

            lines_file = []
            for t, txt_path in enumerate(TXTS_new):
                img_path = txt_path[:-4] + '.jpg'
                line_new = img_path + ' ' + txt_path + ' ' + file_name + ' ' + str(supervised) + ' '  +  '\n'
                lines_file.append(line_new)

            print(f, len(FILES), file_name, len(lines_file))

            lines_all += lines_file
    
        with open(path_txt, 'w') as fid:
            fid.writelines(lines_all)
        
    def process_dev(self, path_data, path_txt, frame_num=8, only_return_lines=False):
        FILES = glob.glob(os.path.join(path_data, '*'))
        print(len(FILES))
        lines_all = []
        for f, file_path in enumerate(FILES):
            file_name = os.path.split(file_path)[-1]
            if self.isInPotocol(file_path):
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
                line_new = img_path + ' ' + txt_path + ' ' + file_name + ' ' + 'dev' + ' '  +  '\n'
                lines_file.append(line_new)

            print(f, len(FILES), file_name, len(lines_file))

            lines_all += lines_file

        if not only_return_lines:
            with open(path_txt, 'w') as fid:
                fid.writelines(lines_all)
        else:
            return lines_all
    
    def process_test(self, path_data, path_txt, frame_num=8, only_return_lines=False):
        FILES = glob.glob(os.path.join(path_data, '*'))
        print(len(FILES))
        lines_all = []
        for f, file_path in enumerate(FILES):
            file_name = os.path.split(file_path)[-1]
            if self.isInPotocol(file_path):
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
        if not only_return_lines:
            with open(path_txt, 'w') as fid:
                fid.writelines(lines_all)
        else:
            return lines_all
