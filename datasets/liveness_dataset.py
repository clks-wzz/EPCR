from __future__ import print_function, division

import torch
import cv2
from PIL import Image
import math
import random

#from ..augmentations.liveness_aug import *
def crop_face_from_scene(image, face_name_full, scale = 1.5):
    #print(face_name_full)
    f=open(face_name_full,'r')
    lines=f.readlines()
    y1,x1,w,h=[float(ele) for ele in lines[:4]]
    f.close()
    y2=y1+w
    x2=x1+h

    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    h_img, w_img = image.shape[0], image.shape[1]
    #w_img,h_img=image.size
    w_scale=scale*w
    h_scale=scale*h
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=max(math.floor(y1),0)
    x1=max(math.floor(x1),0)
    y2=min(math.floor(y2),w_img)
    x2=min(math.floor(x2),h_img)

    #region=image[y1:y2,x1:x2]
    region=image[x1:x2,y1:y2]
    return region

class DatasetOuluP1(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        if self.train:
            self.init_train()
        else:
            self.init_test()

    def init_train(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name.split('_')[-1]
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = int(float(true_label))

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of dataset.'%(self.total_imgs))
    
    def init_test(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name.split('_')[-1]
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = true_label

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of dataset.'%(self.total_imgs))

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        loc_path = self.imgs[img_path]['loc_path']
        label = self.imgs[img_path]['label']
        true_label = self.imgs[img_path]['true_label']
        
        img = cv2.imread(img_path)
        face = crop_face_from_scene(img, loc_path, scale=1.5)
        #print(img, face)

        inputs = self.transform(face)

        if self.train:
            return inputs, label, true_label
        else:
            return inputs, label, true_label, img_path

    def __len__(self):
        return self.total_imgs


class DatasetOuluP1_Extra_RatioAdded(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, train=True, ratio_live=1.0, ratio_spoof=1.0, fully_labeled=False):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        self.ratio_live = ratio_live
        self.ratio_spoof = ratio_spoof
        self.fully_labeled = fully_labeled

        self.filter_users()
        
        if self.train:
            self.init_train()
        else:
            self.init_test()


    def filter_users(self):
        self.users = list(range(1, 21))
        self.users_live = self.users[: int( self.ratio_live * len(self.users) ) ]
        self.users_spoof = self.users[: int( self.ratio_spoof * len(self.users) ) ]

    def init_train(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name.split('_')[-1]
            user_id = int(float(file_name.split('_')[-2]))
            if true_label == '1':
                if label_str == '1':
                    label = 0
                else:
                    label = 1
            else:
                if label_str == '1':
                    if not user_id in self.users_live:
                        continue
                    label = 0
                else:
                    if not user_id in self.users_spoof:
                        continue
                    label = 1
            
            if self.fully_labeled:
                true_label = '1'
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = int(float(true_label))

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of dataset.'%(self.total_imgs))
    
    def init_test(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name.split('_')[-1]
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = true_label

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of dataset.'%(self.total_imgs))

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        loc_path = self.imgs[img_path]['loc_path']
        label = self.imgs[img_path]['label']
        true_label = self.imgs[img_path]['true_label']
        
        img = cv2.imread(img_path)
        face = crop_face_from_scene(img, loc_path, scale=1.5)
        #print(img, face)

        inputs = self.transform(face)

        if self.train:
            return inputs, label, true_label
        else:
            return inputs, label, true_label, img_path

    def __len__(self):
        return self.total_imgs


class DatasetOuluP1_Intra_RatioAdded(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, train=True, ratio_live=1.0, ratio_spoof=1.0, fully_labeled=False):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        self.ratio_live = ratio_live
        self.ratio_spoof = ratio_spoof
        self.fully_labeled = fully_labeled

        self.filter_users()
        
        if self.train:
            self.init_train()
        else:
            self.init_test()


    def filter_users(self):
        self.users = list(range(1, 21))
        self.users_live = self.users[: int( self.ratio_live * len(self.users) ) ]
        self.users_spoof = self.users[: int( self.ratio_spoof * len(self.users) ) ]

    def init_train(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name.split('_')[-1]
            user_id = int(float(file_name.split('_')[-2]))
            if true_label == '1':
                if label_str == '1':
                    label = 0
                else:
                    label = 1
            else:
                continue
                
            if label == 0 and not user_id in self.users_live:
                continue
            if label == 1 and not user_id in self.users_spoof:
                continue
                
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = int(float(true_label))

        # prepare for uniform sampler
        self.index_list_labeled = []
        self.index_list_unlabeled = []
        for i, img_path in enumerate(self.img_list):
            true_label = self.imgs[img_path]['true_label']
            if true_label == 1:
                self.index_list_labeled.append(i)
            else:
                self.index_list_unlabeled.append(i)
        
        print(len(self.index_list_labeled), len(self.index_list_unlabeled))

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of dataset.'%(self.total_imgs))
    
    def init_test(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name.split('_')[-1]
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = true_label

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of dataset.'%(self.total_imgs))

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        loc_path = self.imgs[img_path]['loc_path']
        label = self.imgs[img_path]['label']
        true_label = self.imgs[img_path]['true_label']
        
        img = cv2.imread(img_path)
        face = crop_face_from_scene(img, loc_path, scale=1.5)
        #print(img, face)

        inputs = self.transform(face)

        if self.train:
            return inputs, label, true_label
        else:
            return inputs, label, true_label, img_path

    def __len__(self):
        return self.total_imgs



class DatasetOuluP1_Uniform(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        if self.train:
            self.init_train()
        else:
            self.init_test()

    def init_train(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name.split('_')[-1]
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = int(float(true_label))
        
        # prepare for uniform sampler
        self.index_list_labeled = []
        self.index_list_unlabeled = []
        for i, img_path in enumerate(self.img_list):
            true_label = self.imgs[img_path]['true_label']
            if true_label == 1:
                self.index_list_labeled.append(i)
            else:
                self.index_list_unlabeled.append(i)
        
        print(len(self.index_list_labeled), len(self.index_list_unlabeled))

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of dataset.'%(self.total_imgs))
    
    def init_test(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name.split('_')[-1]
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = true_label

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of dataset.'%(self.total_imgs))

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        loc_path = self.imgs[img_path]['loc_path']
        label = self.imgs[img_path]['label']
        true_label = self.imgs[img_path]['true_label']
        
        img = cv2.imread(img_path)
        face = crop_face_from_scene(img, loc_path, scale=1.5)
        #print(img, face)

        inputs = self.transform(face)

        if self.train:
            return inputs, label, true_label
        else:
            return inputs, label, true_label, img_path

    def __len__(self):
        return self.total_imgs

class DatasetOuluP1_Uniform_Extra_RatioAdded(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, train=True, ratio_live=1.0, ratio_spoof=1.0):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        self.ratio_live = ratio_live
        self.ratio_spoof = ratio_spoof

        self.filter_users()
        
        if self.train:
            self.init_train()
        else:
            self.init_test()


    def filter_users(self):
        self.users = list(range(1, 21))
        self.users_live = self.users[: int( self.ratio_live * len(self.users) ) ]
        self.users_spoof = self.users[: int( self.ratio_spoof * len(self.users) ) ]

    def init_train(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name.split('_')[-1]
            user_id = int(float(file_name.split('_')[-2]))
            if true_label == '1':
                if label_str == '1':
                    label = 0
                else:
                    label = 1
            else:
                if label_str == '1':
                    if not user_id in self.users_live:
                        continue
                    label = 0
                else:
                    if not user_id in self.users_spoof:
                        continue
                    label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = int(float(true_label))

        # prepare for uniform sampler
        self.index_list_labeled = []
        self.index_list_unlabeled = []
        for i, img_path in enumerate(self.img_list):
            true_label = self.imgs[img_path]['true_label']
            if true_label == 1:
                self.index_list_labeled.append(i)
            else:
                self.index_list_unlabeled.append(i)
        
        print(len(self.index_list_labeled), len(self.index_list_unlabeled))

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of dataset.'%(self.total_imgs))
    
    def init_test(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name.split('_')[-1]
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = true_label

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of dataset.'%(self.total_imgs))

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        loc_path = self.imgs[img_path]['loc_path']
        label = self.imgs[img_path]['label']
        true_label = self.imgs[img_path]['true_label']
        
        img = cv2.imread(img_path)
        face = crop_face_from_scene(img, loc_path, scale=1.5)
        #print(img, face)

        inputs = self.transform(face)

        if self.train:
            return inputs, label, true_label
        else:
            return inputs, label, true_label, img_path

    def __len__(self):
        return self.total_imgs


class DatasetOuluP1_Uniform_Intra_RatioAdded(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, train=True, ratio_live=1.0, ratio_spoof=1.0):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        self.ratio_live = ratio_live
        self.ratio_spoof = ratio_spoof

        self.filter_users()
        
        if self.train:
            self.init_train()
        else:
            self.init_test()


    def filter_users(self):
        self.users = list(range(1, 21))
        self.users_live = self.users[: int( self.ratio_live * len(self.users) ) ]
        self.users_spoof = self.users[: int( self.ratio_spoof * len(self.users) ) ]

    def init_train(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name.split('_')[-1]
            user_id = int(float(file_name.split('_')[-2]))
            if true_label == '1':
                if label_str == '1':
                    label = 0
                else:
                    label = 1
            else:
                continue
                
            if label == 0 and not user_id in self.users_live:
                true_label = '0'
            if label == 1 and not user_id in self.users_spoof:
                true_label = '0'
                
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = int(float(true_label))

        # prepare for uniform sampler
        self.index_list_labeled = []
        self.index_list_unlabeled = []
        for i, img_path in enumerate(self.img_list):
            true_label = self.imgs[img_path]['true_label']
            if true_label == 1:
                self.index_list_labeled.append(i)
            else:
                self.index_list_unlabeled.append(i)
        
        print(len(self.index_list_labeled), len(self.index_list_unlabeled))

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of dataset.'%(self.total_imgs))
    
    def init_test(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name.split('_')[-1]
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = true_label

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of dataset.'%(self.total_imgs))

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        loc_path = self.imgs[img_path]['loc_path']
        label = self.imgs[img_path]['label']
        true_label = self.imgs[img_path]['true_label']
        
        img = cv2.imread(img_path)
        face = crop_face_from_scene(img, loc_path, scale=1.5)
        #print(img, face)

        inputs = self.transform(face)

        if self.train:
            return inputs, label, true_label
        else:
            return inputs, label, true_label, img_path

    def __len__(self):
        return self.total_imgs


class DatasetOuluP1_Uniform_Intra_RatioAdded_CrossVerify(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, train=True, ratio_live=1.0, ratio_spoof=1.0, cross_start=0.0):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        self.ratio_live = ratio_live
        self.ratio_spoof = ratio_spoof
        self.cross_start = cross_start

        self.filter_users()
        
        if self.train:
            self.init_train()
        else:
            self.init_test()


    def filter_users(self):
        self.users = list(range(1, 21))
        print('filter users by cross_start:', self.cross_start)
        live_start = int( self.cross_start * len(self.users) )
        live_end = live_start + int( self.ratio_live * len(self.users) )
        spoof_start = int( self.cross_start * len(self.users) )
        spoof_end = spoof_start + int( self.ratio_spoof * len(self.users) )

        self.users_live = self.users[live_start : live_end ]
        self.users_spoof = self.users[spoof_start : spoof_end]

    def init_train(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name.split('_')[-1]
            user_id = int(float(file_name.split('_')[-2]))
            if true_label == '1':
                if label_str == '1':
                    label = 0
                else:
                    label = 1
            else:
                continue
                
            if label == 0 and not user_id in self.users_live:
                true_label = '0'
            if label == 1 and not user_id in self.users_spoof:
                true_label = '0'
                
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = int(float(true_label))

        # prepare for uniform sampler
        self.index_list_labeled = []
        self.index_list_unlabeled = []
        for i, img_path in enumerate(self.img_list):
            true_label = self.imgs[img_path]['true_label']
            if true_label == 1:
                self.index_list_labeled.append(i)
            else:
                self.index_list_unlabeled.append(i)
        
        print(len(self.index_list_labeled), len(self.index_list_unlabeled))

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of dataset.'%(self.total_imgs))
    
    def init_test(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name.split('_')[-1]
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = true_label

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of dataset.'%(self.total_imgs))

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        loc_path = self.imgs[img_path]['loc_path']
        label = self.imgs[img_path]['label']
        true_label = self.imgs[img_path]['true_label']
        
        img = cv2.imread(img_path)
        face = crop_face_from_scene(img, loc_path, scale=1.5)
        #print(img, face)

        inputs = self.transform(face)

        if self.train:
            return inputs, label, true_label
        else:
            return inputs, label, true_label, img_path

    def __len__(self):
        return self.total_imgs

class DatasetOuluP1_OnlyLabeled(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        if self.train:
            self.init_train()
        else:
            self.init_test()

    def init_train(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            if true_label == '0':
                continue
            label_str = file_name.split('_')[-1]
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = int(float(true_label))

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of dataset.'%(self.total_imgs))
    
    def init_test(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name.split('_')[-1]
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = true_label


        self.total_imgs = len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        loc_path = self.imgs[img_path]['loc_path']
        label = self.imgs[img_path]['label']
        true_label = self.imgs[img_path]['true_label']
        
        img = cv2.imread(img_path)
        face = crop_face_from_scene(img, loc_path, scale=1.5)
        #face = crop_face_from_scene(img, loc_path, scale=1.0)
        #print(img, face)

        inputs = self.transform(face)

        if self.train:
            return inputs, label, true_label
        else:
            return inputs, label, true_label, img_path

    def __len__(self):
        return self.total_imgs



class DatasetOuluP1_Oneclass(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        if self.train:
            self.init_train()
        else:
            self.init_test()

    def init_train(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name.split('_')[-1]
            if true_label == '0':
                if label_str == '1':
                    pass
                else:
                    continue
            ##
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = int(float(true_label))

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of dataset.'%(self.total_imgs))
    
    def init_test(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name.split('_')[-1]
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = true_label

        self.total_imgs = len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        loc_path = self.imgs[img_path]['loc_path']
        label = self.imgs[img_path]['label']
        true_label = self.imgs[img_path]['true_label']
        
        img = cv2.imread(img_path)
        face = crop_face_from_scene(img, loc_path, scale=1.5)
        #print(img, face)

        inputs = self.transform(face)

        if self.train:
            return inputs, label, true_label
        else:
            return inputs, label, true_label, img_path

    def __len__(self):
        return self.total_imgs


class DatasetOuluP1_Uniform_Oneclass(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        if self.train:
            self.init_train()
        else:
            self.init_test()

    def init_train(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name.split('_')[-1]
            if true_label == '0':
                if label_str == '1':
                    pass
                    #print('That is one class unlabeled')
                else:
                    continue
            ##
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = int(float(true_label))

        # prepare for uniform sampler
        self.index_list_labeled = []
        self.index_list_unlabeled = []
        for i, img_path in enumerate(self.img_list):
            true_label = self.imgs[img_path]['true_label']
            if true_label == 1:
                self.index_list_labeled.append(i)
            else:
                self.index_list_unlabeled.append(i)
        
        print(len(self.index_list_labeled), len(self.index_list_unlabeled))

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of training dataset.'%(self.total_imgs))
        print('unlabled:%d'%(len(self.index_list_unlabeled)))
        print('labled:%d'%(len(self.index_list_labeled)))

    
    def init_test(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name.split('_')[-1]
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = true_label

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of testing dataset.'%(self.total_imgs))

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        loc_path = self.imgs[img_path]['loc_path']
        label = self.imgs[img_path]['label']
        true_label = self.imgs[img_path]['true_label']
        
        img = cv2.imread(img_path)
        face = crop_face_from_scene(img, loc_path, scale=1.5)
        #print(img, face)

        inputs = self.transform(face)

        if self.train:
            return inputs, label, true_label
        else:
            return inputs, label, true_label, img_path

    def __len__(self):
        return self.total_imgs


class DatasetOCIM(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        if self.train:
            self.init_train()
        else:
            self.init_test()

    def init_train(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = int(float(true_label))

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of train dataset.'%(self.total_imgs))
    
    def init_test(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = true_label

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of test dataset.'%(self.total_imgs))

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        loc_path = self.imgs[img_path]['loc_path']
        label = self.imgs[img_path]['label']
        true_label = self.imgs[img_path]['true_label']
        
        img = cv2.imread(img_path)
        face = crop_face_from_scene(img, loc_path, scale=1.5)
        #print(img, face)

        inputs = self.transform(face)

        if self.train:
            return inputs, label, true_label
        else:
            return inputs, label, true_label, img_path

    def __len__(self):
        return self.total_imgs



    
class DatasetOCIM_OnlyLabeled(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        if self.train:
            self.init_train()
        else:
            self.init_test()

    def init_train(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            if true_label == '0':
                continue
            label_str = file_name
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = int(float(true_label))

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of dataset.'%(self.total_imgs))
    
    def init_test(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = true_label

        self.total_imgs = len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        loc_path = self.imgs[img_path]['loc_path']
        label = self.imgs[img_path]['label']
        true_label = self.imgs[img_path]['true_label']
        
        img = cv2.imread(img_path)
        if img is None:
            print(img_path)
        face = crop_face_from_scene(img, loc_path, scale=1.5)
        #print(img, face)

        inputs = self.transform(face)

        if self.train:
            return inputs, label, true_label
        else:
            return inputs, label, true_label, img_path

    def __len__(self):
        return self.total_imgs



class DatasetOCIM_Uniform(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        if self.train:
            self.init_train()
        else:
            self.init_test()

    def init_train(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = int(float(true_label))
        
        # prepare for uniform sampler
        self.index_list_labeled = []
        self.index_list_unlabeled = []
        for i, img_path in enumerate(self.img_list):
            true_label = self.imgs[img_path]['true_label']
            if true_label == 1:
                self.index_list_labeled.append(i)
            else:
                self.index_list_unlabeled.append(i)

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of train dataset.'%(self.total_imgs))
    
    def init_test(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = true_label

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of test dataset.'%(self.total_imgs))

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        loc_path = self.imgs[img_path]['loc_path']
        label = self.imgs[img_path]['label']
        true_label = self.imgs[img_path]['true_label']
        
        img = cv2.imread(img_path)
        face = crop_face_from_scene(img, loc_path, scale=1.5)
        #print(img, face)

        inputs = self.transform(face)

        if self.train:
            return inputs, label, true_label
        else:
            return inputs, label, true_label, img_path

    def __len__(self):
        return self.total_imgs

def max_lists(lists):
    for _list in lists:
        _max = 0
        for _num in _list:
            _max = max(_max, _num)
        print(_max)


class DatasetOCIM_SSDG_Uniform(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        if self.train:
            self.init_train()
        else:
            self.init_test()

    def get_indexes(self):
        return [self.idx_domain1_real, self.idx_domain1_fake, self.idx_domain2_real, self.idx_domain2_fake, self.idx_domain3_real, self.idx_domain3_fake]

    def init_train(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}

        self.idx_domain1_real = []
        self.idx_domain1_fake = []
        self.idx_domain2_real = []
        self.idx_domain2_fake = []
        self.idx_domain3_real = []
        self.idx_domain3_fake = []
        
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, domain, true_label = line.strip().split()
            label_str = file_name
            if label_str == '1':
                label = 0
            else:
                label = 1
            
            if domain == '0':
                if label_str == '1':
                    self.idx_domain1_real.append(l)
                else:
                    self.idx_domain1_fake.append(l)
            elif domain == '1':
                if label_str == '1':
                    self.idx_domain2_real.append(l)
                else:
                    self.idx_domain2_fake.append(l)
            elif domain == '2':
                if label_str == '1':
                    self.idx_domain3_real.append(l)
                else:
                    self.idx_domain3_fake.append(l)
            else:
                raise ValueError

            domain = int(float(domain))

            if label_str == '1':
                triplet = 0
            else:
                triplet = domain + 1

            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['domain'] = domain
            self.imgs[img_path]['triplet'] = triplet
            self.imgs[img_path]['true_label'] = int(float(true_label))
        
        print(len(self.img_list))
        max_lists(self.get_indexes())
        
        self.total_imgs = len(self.img_list)
        print('There are %d imgs of train dataset.'%(self.total_imgs))
    
    def init_test(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = true_label

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of test dataset.'%(self.total_imgs))

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        loc_path = self.imgs[img_path]['loc_path']
        label = self.imgs[img_path]['label']
        true_label = self.imgs[img_path]['true_label']
        
        img = cv2.imread(img_path)
        face = crop_face_from_scene(img, loc_path, scale=1.5)
        #print(img, face)

        inputs = self.transform(face)

        if self.train:
            domain = self.imgs[img_path]['domain']
            triplet = self.imgs[img_path]['triplet']
            return inputs, label, domain, triplet
        else:
            return inputs, label, true_label, img_path

    def __len__(self):
        return self.total_imgs


class DatasetOCIM_SSDG_Sparse_Uniform(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, train=True, sample_num=3, crop_scale=1.5):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        self.sample_num = sample_num
        self.crop_scale = crop_scale
        print('crop_scale:', crop_scale, self.crop_scale)
        if self.train:
            self.init_train()
        else:
            self.init_test()

    def get_indexes(self):
        return [self.idx_domain1_real, self.idx_domain1_fake, self.idx_domain2_real, self.idx_domain2_fake, self.idx_domain3_real, self.idx_domain3_fake]

    def get_samples_avg(self, _list, _sample_num):
        if len(_list) <= _sample_num:
            return _list
        _interval = len(_list) // _sample_num
        res_list = []
        for i in range(0, len(_list), _interval):
            res_list.append(_list[i])
        
        return res_list

    def init_train(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        dicts = {}
        for l, line in enumerate(lines):
            img_path, dat_path, _, domain, true_label = line.strip().split()
            dir_name = img_path.split('/')[-2]
            dicts.setdefault(dir_name, [])
            dicts[dir_name].append(line)
        
        lines_new = []
        for dir_name in dicts.keys():
            dicts[dir_name] = self.get_samples_avg(dicts[dir_name], self.sample_num)
            lines_new += dicts[dir_name]

        lines = lines_new
        
        self.img_list = []
        self.imgs = {}

        self.idx_domain1_real = []
        self.idx_domain1_fake = []
        self.idx_domain2_real = []
        self.idx_domain2_fake = []
        self.idx_domain3_real = []
        self.idx_domain3_fake = []
        
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, domain, true_label = line.strip().split()
            label_str = file_name
            if label_str == '1':
                label = 0
            else:
                label = 1
            
            if domain == '0':
                if label_str == '1':
                    self.idx_domain1_real.append(l)
                else:
                    self.idx_domain1_fake.append(l)
            elif domain == '1':
                if label_str == '1':
                    self.idx_domain2_real.append(l)
                else:
                    self.idx_domain2_fake.append(l)
            elif domain == '2':
                if label_str == '1':
                    self.idx_domain3_real.append(l)
                else:
                    self.idx_domain3_fake.append(l)
            else:
                raise ValueError

            domain = int(float(domain))

            if label_str == '1':
                triplet = 0
            else:
                triplet = domain + 1

            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['domain'] = domain
            self.imgs[img_path]['triplet'] = triplet
            self.imgs[img_path]['true_label'] = int(float(true_label))
        
        print(len(self.img_list))
        max_lists(self.get_indexes())
        
        self.total_imgs = len(self.img_list)
        print('There are %d imgs of train dataset.'%(self.total_imgs))
    
    def init_test(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = true_label

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of test dataset.'%(self.total_imgs))

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        loc_path = self.imgs[img_path]['loc_path']
        label = self.imgs[img_path]['label']
        true_label = self.imgs[img_path]['true_label']
        
        img = cv2.imread(img_path)
        face = crop_face_from_scene(img, loc_path, scale=self.crop_scale)
        #print(img, face)

        inputs = self.transform(face)

        if self.train:
            domain = self.imgs[img_path]['domain']
            triplet = self.imgs[img_path]['triplet']
            return inputs, label, domain, triplet
        else:
            return inputs, label, true_label, img_path

    def __len__(self):
        return self.total_imgs



class DatasetOCIM_Uniform_Extra_RatioAdded(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, train=True, ratio_live=1.0, ratio_spoof=1.0):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        self.ratio_live = ratio_live
        self.ratio_spoof = ratio_spoof
        self.filter_users()

        if self.train:
            self.init_train()
        else:
            self.init_test()
    
    def filter_users(self):
        self.users = list(range(1, 21))
        self.users_live = self.users[: int( self.ratio_live * len(self.users) ) ]
        self.users_spoof = self.users[: int( self.ratio_spoof * len(self.users) ) ]

    def init_train(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name
            video_name = img_path.split('/')[-2]

            if true_label == '1':
                if label_str == '1':
                    label = 0
                else:
                    label = 1
            else:
                user_id = int(float(video_name.split('_')[2]))
                assert('CASIA' in video_name)
                if label_str == '1':
                    if not user_id in self.users_live:
                        continue
                    label = 0
                else:
                    if not user_id in self.users_spoof:
                        continue
                    label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = int(float(true_label))
        
        # prepare for uniform sampler
        self.index_list_labeled = []
        self.index_list_unlabeled = []
        for i, img_path in enumerate(self.img_list):
            true_label = self.imgs[img_path]['true_label']
            if true_label == 1:
                self.index_list_labeled.append(i)
            else:
                self.index_list_unlabeled.append(i)

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of train dataset.'%(self.total_imgs))
    
    def init_test(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = true_label

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of test dataset.'%(self.total_imgs))

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        loc_path = self.imgs[img_path]['loc_path']
        label = self.imgs[img_path]['label']
        true_label = self.imgs[img_path]['true_label']
        
        img = cv2.imread(img_path)
        face = crop_face_from_scene(img, loc_path, scale=1.5)
        #print(img, face)

        inputs = self.transform(face)

        if self.train:
            return inputs, label, true_label
        else:
            return inputs, label, true_label, img_path

    def __len__(self):
        return self.total_imgs



class DatasetOCIM_SupervisedOnly_Intra_RatioAdded(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, train=True, ratio_live=1.0, ratio_spoof=1.0):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        self.ratio_live = ratio_live
        self.ratio_spoof = ratio_spoof
        self.filter_users()

        if self.train:
            self.init_train()
        else:
            self.init_test()
    
    def filter_users(self):
        self.users_O = list(range(1, 21))
        self.users_live_O = self.users_O[: int( self.ratio_live * len(self.users_O) ) ]
        self.users_spoof_O = self.users_O[: int( self.ratio_spoof * len(self.users_O) ) ]

        self.users_C = list(range(1, 21))
        self.users_live_C = self.users_C[: int( self.ratio_live * len(self.users_C) ) ]
        self.users_spoof_C = self.users_C[: int( self.ratio_spoof * len(self.users_C) ) ]

        self.users_I = [1, 2, 4, 6, 7, 8, 12, 16, 18, 25, 27, 103, 105, 108, 110]
        self.users_live_I = self.users_I[: int( self.ratio_live * len(self.users_I) ) ]
        self.users_spoof_I = self.users_I[: int( self.ratio_spoof * len(self.users_I) ) ]

        self.users_M = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 21, 22, 23, 24, 26, 28, 29, 30, 32, 33, 34, 35, 36, 37, 39, 42, 48, 49, 50, 51, 53, 54, 55]
        self.users_live_M = self.users_M[: int( self.ratio_live * len(self.users_M) ) ]
        self.users_spoof_M = self.users_M[: int( self.ratio_spoof * len(self.users_M) ) ]

    def check_true_label(self, img_path, label_str):
        video_name = img_path.split('/')[-2]

        if 'CASIA' in video_name:
            user_id = int(float(video_name.split('_')[2]))
            if label_str == '1':
                if user_id in self.users_live_C:
                    return 1
                else:
                    return 0
            else:
                if user_id in self.users_spoof_C:
                    return 1
                else:
                    return 0
        elif 'ReplayAttack' in video_name:
            user_id = int(float(video_name.split('_')[-3]))
            if label_str == '1':
                if user_id in self.users_live_I:
                    return 1
                else:
                    return 0
            else:
                if user_id in self.users_spoof_I:
                    return 1
                else:
                    return 0
        elif 'MSU_imgs' in img_path:
            pid = video_name.split('_')[1][len('client'):]
            user_id = int(float(pid))
            if label_str == '1':
                if user_id in self.users_live_M:
                    return 1
                else:
                    return 0
            else:
                if user_id in self.users_spoof_M:
                    return 1
                else:
                    return 0
        elif 'OULU' in img_path:
            user_id = int(float(video_name.split('_')[2]))
            if label_str == '1':
                if user_id in self.users_live_O:
                    return 1
                else:
                    return 0
            else:
                if user_id in self.users_spoof_O:
                    return 1
                else:
                    return 0
        else:
            raise ValueError

    def init_train(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name

            if label_str == '1':
                label = 0
            else:
                label = 1

            true_label = self.check_true_label(img_path, label_str)

            # remove unlabeled data
            if true_label == 1:
                pass
            else:
                continue


            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = int(float(true_label))
        
        # prepare for uniform sampler
        self.index_list_labeled = []
        self.index_list_unlabeled = []
        for i, img_path in enumerate(self.img_list):
            true_label = self.imgs[img_path]['true_label']
            if true_label == 1:
                self.index_list_labeled.append(i)
            else:
                self.index_list_unlabeled.append(i)

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of train dataset.'%(self.total_imgs))
    
    def init_test(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = true_label

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of test dataset.'%(self.total_imgs))

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        loc_path = self.imgs[img_path]['loc_path']
        label = self.imgs[img_path]['label']
        true_label = self.imgs[img_path]['true_label']
        
        img = cv2.imread(img_path)
        face = crop_face_from_scene(img, loc_path, scale=1.5)
        #print(img, face)

        inputs = self.transform(face)

        if self.train:
            return inputs, label, true_label
        else:
            return inputs, label, true_label, img_path

    def __len__(self):
        return self.total_imgs

class DatasetOCIM_Uniform_Intra_RatioAdded(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, train=True, ratio_live=1.0, ratio_spoof=1.0):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        self.ratio_live = ratio_live
        self.ratio_spoof = ratio_spoof
        self.filter_users()

        if self.train:
            self.init_train()
        else:
            self.init_test()
    
    def filter_users(self):
        self.users_O = list(range(1, 21))
        self.users_live_O = self.users_O[: int( self.ratio_live * len(self.users_O) ) ]
        self.users_spoof_O = self.users_O[: int( self.ratio_spoof * len(self.users_O) ) ]

        self.users_C = list(range(1, 21))
        self.users_live_C = self.users_C[: int( self.ratio_live * len(self.users_C) ) ]
        self.users_spoof_C = self.users_C[: int( self.ratio_spoof * len(self.users_C) ) ]

        self.users_I = [1, 2, 4, 6, 7, 8, 12, 16, 18, 25, 27, 103, 105, 108, 110]
        self.users_live_I = self.users_I[: int( self.ratio_live * len(self.users_I) ) ]
        self.users_spoof_I = self.users_I[: int( self.ratio_spoof * len(self.users_I) ) ]

        self.users_M = [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 21, 22, 23, 24, 26, 28, 29, 30, 32, 33, 34, 35, 36, 37, 39, 42, 48, 49, 50, 51, 53, 54, 55]
        self.users_live_M = self.users_M[: int( self.ratio_live * len(self.users_M) ) ]
        self.users_spoof_M = self.users_M[: int( self.ratio_spoof * len(self.users_M) ) ]

    def check_true_label(self, img_path, label_str):
        video_name = img_path.split('/')[-2]

        if 'CASIA' in video_name:
            user_id = int(float(video_name.split('_')[2]))
            if label_str == '1':
                if user_id in self.users_live_C:
                    return 1
                else:
                    return 0
            else:
                if user_id in self.users_spoof_C:
                    return 1
                else:
                    return 0
        elif 'ReplayAttack' in video_name:
            user_id = int(float(video_name.split('_')[-3]))
            if label_str == '1':
                if user_id in self.users_live_I:
                    return 1
                else:
                    return 0
            else:
                if user_id in self.users_spoof_I:
                    return 1
                else:
                    return 0
        elif 'MSU_imgs' in img_path:
            pid = video_name.split('_')[1][len('client'):]
            user_id = int(float(pid))
            if label_str == '1':
                if user_id in self.users_live_M:
                    return 1
                else:
                    return 0
            else:
                if user_id in self.users_spoof_M:
                    return 1
                else:
                    return 0
        elif 'OULU' in img_path:
            user_id = int(float(video_name.split('_')[2]))
            if label_str == '1':
                if user_id in self.users_live_O:
                    return 1
                else:
                    return 0
            else:
                if user_id in self.users_spoof_O:
                    return 1
                else:
                    return 0
        else:
            raise ValueError

    def init_train(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name

            if label_str == '1':
                label = 0
            else:
                label = 1

            true_label = self.check_true_label(img_path, label_str)


            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = int(float(true_label))
        
        # prepare for uniform sampler
        self.index_list_labeled = []
        self.index_list_unlabeled = []
        for i, img_path in enumerate(self.img_list):
            true_label = self.imgs[img_path]['true_label']
            if true_label == 1:
                self.index_list_labeled.append(i)
            else:
                self.index_list_unlabeled.append(i)

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of train dataset.'%(self.total_imgs))
    
    def init_test(self):
        with open(self.data_dir, 'r') as fid:
            lines = fid.readlines()
        
        self.img_list = []
        self.imgs = {}
        for l, line in enumerate(lines):
            img_path, dat_path, file_name, true_label = line.strip().split()
            label_str = file_name
            if label_str == '1':
                label = 0
            else:
                label = 1
            self.img_list.append(img_path)
            self.imgs.setdefault(img_path, {})
            self.imgs[img_path]['loc_path'] = dat_path
            self.imgs[img_path]['label'] = label
            self.imgs[img_path]['true_label'] = true_label

        self.total_imgs = len(self.img_list)
        print('There are %d imgs of test dataset.'%(self.total_imgs))

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        loc_path = self.imgs[img_path]['loc_path']
        label = self.imgs[img_path]['label']
        true_label = self.imgs[img_path]['true_label']
        
        img = cv2.imread(img_path)
        face = crop_face_from_scene(img, loc_path, scale=1.5)
        #print(img, face)

        inputs = self.transform(face)

        if self.train:
            return inputs, label, true_label
        else:
            return inputs, label, true_label, img_path

    def __len__(self):
        return self.total_imgs


class BatchSampler_Uniform(torch.utils.data.Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices."""
    def __init__(self, dataset, batch_size, drop_last=True, labeled_ratio=0.8):
        assert(drop_last)

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.index_list_labeled = dataset.index_list_labeled
        self.index_list_unlabeled = dataset.index_list_unlabeled

        self.batch_size_labeled = int(batch_size * labeled_ratio) 
        self.batch_size_unlabeled = batch_size - self.batch_size_labeled

        #self.batch_num_per_epoch = len(self.index_list_labeled) // self.batch_size_labeled
        if self.drop_last:
            self.batch_num_per_epoch = len(self.index_list_labeled) // self.batch_size_labeled
        else:
            self.batch_num_per_epoch = (len(self.index_list_labeled) + self.batch_size_labeled - 1) // self.batch_size_labeled

    def __iter__(self):
        '''
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []   
        if len(batch) > 0 and not self.drop_last:
            yield batch
        '''
        print(len(self.index_list_labeled), len(self.index_list_unlabeled))
        if len(self.index_list_unlabeled) > len(self.index_list_labeled):
            pass
        else:
            self.index_list_unlabeled = self.index_list_unlabeled * int(len(self.index_list_labeled) // len(self.index_list_unlabeled) + 1)
        random.shuffle(self.index_list_labeled)
        random.shuffle(self.index_list_unlabeled)

        batch = []
        idx_labeled = 0
        idx_unlabeled = 0
        for b in range(self.batch_num_per_epoch):
            for i in range(self.batch_size_labeled):
                batch.append(self.index_list_labeled[idx_labeled])
                idx_labeled += 1
            for i in range(self.batch_size_unlabeled):
                batch.append(self.index_list_unlabeled[idx_unlabeled])
                idx_unlabeled += 1
            yield batch
            batch = []

    def __len__(self):
        if self.drop_last:
            return len(self.index_list_labeled) // self.batch_size_labeled
        else:
            return (len(self.index_list_labeled) + self.batch_size_labeled - 1) // self.batch_size_labeled



class BatchSampler_SSDG_Uniform(torch.utils.data.Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices."""
    def __init__(self, dataset, batch_size, drop_last=True, labeled_ratio=0.8):
        assert(drop_last)

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        #self.idx_domain1_real, self.idx_domain1_fake, self.idx_domain2_real, self.idx_domain2_fake, self.idx_domain3_real, self.idx_domain3_fake = self.dataset.get_indexes()
        self.indexes_lists = self.dataset.get_indexes()

        max_len = self.get_max_length(self.indexes_lists)
        self.max_len = max_len

        self.batch_num_per_epoch = max_len // self.batch_size
        self.batch_size_per_ds = self.batch_size // len(self.indexes_lists)

        assert(self.batch_size % len(self.indexes_lists) == 0)
        
        for i, indexes_list in enumerate(self.indexes_lists):
            self.indexes_lists[i] = indexes_list * int(self.max_len // len(indexes_list) + 2)
            #random.shuffle(self.indexes_lists[i])
        

    def random_shuffle(self, indexes_lists):
        new_lists = []
        for indexes_list in indexes_lists:
            random.shuffle(indexes_list)
    
    def get_max_length(self, indexes_lists):
        max_len = 0
        for indexes_list in indexes_lists:
            max_len = max(max_len, len(indexes_list))
        return max_len

    def __iter__(self):
        #self.idx_domain1_real, self.idx_domain1_fake, self.idx_domain2_real, self.idx_domain2_fake, self.idx_domain3_real, self.idx_domain3_fake = self.random_shuffle([self.idx_domain1_real, self.idx_domain1_fake, self.idx_domain2_real, self.idx_domain2_fake, self.idx_domain3_real, self.idx_domain3_fake])
        #indexes_lists = self.random_shuffle(indexes_lists)

        for i, indexes_list in enumerate(self.indexes_lists):
            #self.indexes_lists[i] = indexes_list * int(self.max_len // len(indexes_list) + 2)
            random.shuffle(self.indexes_lists[i])
        
        batch = []
        idx_all_ds = [0 for i in range(len(self.indexes_lists))]
        for b in range(self.batch_num_per_epoch):
            for i, indexes_list in enumerate(self.indexes_lists):
                for j in range(self.batch_size_per_ds):
                    this_idx = idx_all_ds[i]
                    batch.append(indexes_list[this_idx])
                    idx_all_ds[i] += 1

            yield batch
            batch = []

    def __len__(self):
        return self.batch_num_per_epoch
