from .liveness_aug import *

def get_aug(name, image_size, train, debug=False, augment_choices=None, train_classifier=True):

    if True:
        if name in [ 'simsiam_semi_cdcn_meanteacherV11_ssdg_fp16']:
            augmentation = SimSiamSemiTransform(image_size, train, debug=debug)
        else:
            raise NotImplementedError
    #else:
    #    augmentation = Transform_single(image_size, train=train_classifier)

    return augmentation








