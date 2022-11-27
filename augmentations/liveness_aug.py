from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 
import imgaug.augmenters as iaa

import numpy as np
import numbers
import types
import collections
import warnings
import cv2
import sys
from PIL import Image, ImageFile, ImageEnhance

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

#face_scale = 1.3  #default for test, for training , can be set from [1.2 to 1.5]

# data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
seq = iaa.Sequential([
    iaa.Add(value=(-40,40), per_channel=True), # Add color 
    iaa.GammaContrast(gamma=(0.5,1.5)) # GammaContrast with a gamma of 0.5 to 1.5
])

gaussian_blur = iaa.Sequential([
    iaa.GaussianBlur(sigma=(0.0, 5.0))
])
 

def crop_face_from_scene(image, face_name_full, scale = 1.5):
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


def rotate_image(im, angle):
    r"""rotation image"""
    if angle % 90 == 0:
        angle = angle % 360
        if angle == 0:
            return im
        elif angle == 90:
            return im.transpose((1, 0, 2))[:, ::-1, :]
        elif angle == 180:
            return im[::-1, ::-1, :]
        elif angle == 270:
            return im.transpose((1, 0, 2))[::-1, :, :]
    else:
        raise Exception('Error')


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy(img):
    return isinstance(img, np.ndarray)


def _is_numpy_image(img):
    return img.ndim in {2, 3}

def crop(img, i, j, h, w):
    if not _is_numpy_image(img):
        raise TypeError('img should be OpenCV numpy Image. Got {}'.format(type(img)))
    return img[i:i+h, j:j+w, :]

def resize(img, size, interpolation=cv2.INTER_LINEAR):
    if not _is_numpy_image(img):
        raise TypeError('img should be OpenCV numpy Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        h, w, _ = img.shape
        #h, w= img.shape
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(img, (ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(img, (ow, oh), interpolation)
    else:
        return cv2.resize(img, size[::-1], interpolation)



# array
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.01, sh = 0.05, r1 = 0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, sample):
        img = sample
        
        if random.uniform(0, 1) < self.probability:
            attempts = np.random.randint(1, 3)
            for attempt in range(attempts):
                area = img.shape[0] * img.shape[1]
           
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1/self.r1)
    
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
    
                if w < img.shape[1] and h < img.shape[0]:
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)

                    img[x1:x1+h, y1:y1+w, 0] = self.mean[0]
                    img[x1:x1+h, y1:y1+w, 1] = self.mean[1]
                    img[x1:x1+h, y1:y1+w, 2] = self.mean[2]
                    
        return img


# Tensor
class Cutout(object):
    def __init__(self, length=50):
        self.length = length

    def __call__(self, sample):
        img = sample
        h, w = img.shape[1], img.shape[2]    # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_new = np.random.randint(1, self.length)
        
        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x = sample
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        return new_image_x

class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""
    def __call__(self, sample):
        image_x = sample

        p = random.random()
        if p < 0.5:
            #print('Flip')
            new_image_x = cv2.flip(image_x, 1)

            return new_image_x
        else:
            #print('no Flip')
            return image_x

class RandomRotation(object):
    """Rotate the image randomly"""
    def __call__(self, sample):
        image_x = sample

        angle = random.sample([90, 180, 270], 1)[0]
        
        '''
        p = random.random()
        if p < 0.5:
            return rotate_image(image_x, angle)
        else:
            return image_x
        '''
        return rotate_image(image_x, angle)

class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image = sample
        image = image[:,:,::-1].transpose((2, 0, 1))
        image = torch.from_numpy(image.astype(np.float)).float()
        
        return image


class Resize(object):
    """Resize the input numpy ndarray to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_CUBIC``, bicubic interpolation
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be scaled.
        Returns:
            numpy ndarray: Rescaled image.
        """
        return resize(img, self.size, self.interpolation)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, self.interpolation)


class RandomResizedCrop(object):
    """Crop the given numpy ndarray to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: cv2.INTER_CUBIC
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=cv2.INTER_LINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (numpy ndarray): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.shape[1] and h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w
 

    def resized_crop(self, img, i, j, h, w, size, interpolation=cv2.INTER_LINEAR):
        assert _is_numpy_image(img), 'img should be OpenCV numpy Image'
        img = crop(img, i, j, h, w)
        img = resize(img, size, interpolation)
        return img

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be cropped and resized.
        Returns:
            numpy ndarray: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return self.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        #interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        #format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        #format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        #format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class RandomShufflePatch(object):
    def __init__(self, image_size, ratio=0.5, total_patch_num=9): 
        self.ratio = ratio
        self.total_patch_num = total_patch_num
        self.patch_num = int(math.sqrt(self.total_patch_num))
        self.image_size = image_size
        self.patch_size = image_size // self.patch_num
        self.w_list, self.h_list = [], []
        for i in range(self.patch_num):
            if i == self.patch_num - 1:
                self.w_list.append([i * self.patch_size, image_size])
                self.h_list.append([i * self.patch_size, image_size])
            else:
                self.w_list.append([i * self.patch_size, (i+1) * self.patch_size])
                self.h_list.append([i * self.patch_size, (i+1) * self.patch_size])
        #print('RandomShufflePatch')
        
    def __call__(self, img):
        #print('RandomShufflePatch write img')
        h_list = self.h_list
        w_list = self.w_list
        if random.random() < self.ratio:
            return img

        shape = img.shape
        assert shape[0] == shape[1]
        assert shape[0] == self.image_size

        self.patches = []
        for i in range(self.patch_num):
            for j in range(self.patch_num):
                img_patch = img[h_list[i][0]:h_list[i][1], w_list[j][0]:w_list[j][1]]
                img_patch = resize(img_patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
                self.patches.append(img_patch)
        
        random.shuffle(self.patches)
        self.patches_2 = []
        for i in range(self.patch_num):
            self.patches_2.append(np.concatenate(self.patches[i*self.patch_num : (i+1) * self.patch_num], axis=1))
        
        shuffled_img = np.concatenate(self.patches_2, axis=0)

        shuffled_img = resize(shuffled_img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        
        #cv2.imwrite('shuffled_patch.jpg', shuffled_img)

        return shuffled_img

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string



class SimSiamSemiTransform_Selection():
    def __init__(self, image_size, train=True, augment_choices=None):
        self.train = train
        assert(augment_choices is not None)

        trans_list = self.select_choices(augment_choices, image_size)

        if self.train:
            
            self.transform = transforms.Compose(
                trans_list
            )

            print(self.transform)
            
        else:
            self.transform = transforms.Compose([
                Resize((image_size, image_size)),
                ToTensor(),
                Normaliztion()
            ])

    def select_choices(self, choices_str, image_size):
        choices = [float(e) for e in choices_str[1:-1].split(',')]
        trans_list = []

        for _choice in choices:
            assert( _choice in [0., 1., 2., 3., 4., 5.])
                

        if 0 in choices:
            trans_list.append(RandomResizedCrop(image_size, scale=(0.2, 1.0)))
        else:
            trans_list.append(Resize((image_size, image_size)))


        if 1 in choices:
            trans_list.append(gaussian_blur.augment_image)
        if 2 in choices:
            trans_list.append(RandomShufflePatch(image_size))
        if 3 in choices:
            trans_list.append(seq.augment_image)
        if 4 in choices:
            trans_list.append(RandomErasing())
        if 5 in choices:
            trans_list.append(RandomHorizontalFlip())
        
        trans_list.append(ToTensor())

        if 4 in choices:
            trans_list.append(Cutout())
        
        trans_list.append(Normaliztion())
        
        #print('Selecting transfroms:')
        #for _t, _tran in enumerate(trans_list):
        #    print(_t, _tran.__name__)

        return trans_list


    def __call__(self, x):
        if self.train:
            x1 = self.transform(x)
            x2 = self.transform(x)
            return x1, x2 
        else:
            x1 = self.transform(x)
            return x1

# [RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]
class SimSiamSemiTransform():
    def __init__(self, image_size, train=True, debug=False):
        self.train = train
        self.debug = debug
        if self.train:
            '''
            # aug_v0
            self.transform = transforms.Compose([
                RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                seq.augment_image, 
                RandomErasing(),
                RandomHorizontalFlip(),
                ToTensor(),
                Cutout(),
                Normaliztion()
            ])
            
            # aug_v1 remove seq.augment_image
            self.transform = transforms.Compose([
                RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                RandomErasing(),
                RandomHorizontalFlip(),
                ToTensor(),
                Cutout(),
                Normaliztion()
            ])
            
            # aug_v2 
            self.transform = transforms.Compose([
                RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                RandomHorizontalFlip(),
                ToTensor(),
                Normaliztion()
            ])
            '''
            # aug_v5
            # print('aug_v5', RandomShufflePatch(image_size))
            self.transform = transforms.Compose([
                RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                RandomShufflePatch(image_size),
                seq.augment_image, 
                RandomErasing(),
                RandomHorizontalFlip(),
                ToTensor(),
                Cutout(),
                Normaliztion()
            ])
            
        else:
            self.transform = transforms.Compose([
                Resize((image_size, image_size)),
                ToTensor(),
                Normaliztion()
            ])

        if self.debug:
            self.transform_train = transforms.Compose([
                RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                RandomShufflePatch(image_size),
                seq.augment_image, 
                RandomErasing(),
                RandomHorizontalFlip(),
                ToTensor(),
                Cutout(),
                Normaliztion()
            ])
            self.transform_test = transforms.Compose([
                Resize((image_size, image_size)),
                ToTensor(),
                Normaliztion()
            ])

    def __call__(self, x):
        if self.debug:
            #x1 = self.transform_train(x)
            x1 = self.transform_test(x)
            x2 = self.transform_train(x)
            #x2 = self.transform_test(x)
            return x1, x2 

        if self.train:
            x1 = self.transform(x)
            x2 = self.transform(x)
            return x1, x2 
        else:
            x1 = self.transform(x)
            return x1



# [RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]
class SimSiamSemiTransform_Rotation():
    def __init__(self, image_size, train=True, debug=False):
        self.train = train
        self.debug = debug
        if self.train:
            '''
            # aug_v0
            self.transform = transforms.Compose([
                RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                seq.augment_image, 
                RandomErasing(),
                RandomHorizontalFlip(),
                ToTensor(),
                Cutout(),
                Normaliztion()
            ])
            
            # aug_v1 remove seq.augment_image
            self.transform = transforms.Compose([
                RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                RandomErasing(),
                RandomHorizontalFlip(),
                ToTensor(),
                Cutout(),
                Normaliztion()
            ])
            
            # aug_v2 
            self.transform = transforms.Compose([
                RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                RandomHorizontalFlip(),
                ToTensor(),
                Normaliztion()
            ])
            '''
            # aug_v5
            # print('aug_v5', RandomShufflePatch(image_size))
            self.transform = transforms.Compose([
                RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                RandomShufflePatch(image_size),
                seq.augment_image, 
                RandomErasing(),
                RandomHorizontalFlip(),
                RandomRotation(),
                ToTensor(),
                Cutout(),
                Normaliztion()
            ])
            
        else:
            self.transform = transforms.Compose([
                Resize((image_size, image_size)),
                ToTensor(),
                Normaliztion()
            ])

        if self.debug:
            self.transform_train = transforms.Compose([
                RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                RandomShufflePatch(image_size),
                seq.augment_image, 
                RandomErasing(),
                RandomHorizontalFlip(),
                RandomRotation(),
                ToTensor(),
                Cutout(),
                Normaliztion()
            ])
            self.transform_test = transforms.Compose([
                Resize((image_size, image_size)),
                ToTensor(),
                Normaliztion()
            ])

    def __call__(self, x):
        if self.debug:
            #x1 = self.transform_train(x)
            x1 = self.transform_test(x)
            x2 = self.transform_train(x)
            #x2 = self.transform_test(x)
            return x1, x2 

        if self.train:
            x1 = self.transform(x)
            x2 = self.transform(x)
            return x1, x2 
        else:
            x1 = self.transform(x)
            return x1
        