import argparse
import os
import torch

import numpy as np
import torch
import random



def set_deterministic(seed):
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 
    else:
        print("Non-deterministic")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--amp', action='store_true', help='choose FP16 to train?')
    # training specific args
    parser.add_argument('--dataset', type=str, default='cifar10', help='choose from random, stl10, mnist, cifar10, cifar100, imagenet')
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default=os.getenv('DATA'))
    parser.add_argument('--test_dir', type=str, default=os.getenv('TEST'))
    parser.add_argument('--output_dir', type=str, default=os.getenv('OUTPUT'))
    parser.add_argument('--visualize_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--resume_from_breakpoint', action='store_true', help='resume from breakpoint?')
    parser.add_argument('--eval_from', type=str, default=None)

    parser.add_argument('--hide_progress', action='store_true')
    parser.add_argument('--use_default_hyperparameters', action='store_true')
    # model related params
    parser.add_argument('--model', type=str, default='simsiam')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--num_epochs', type=int, default=100, help='This will affect learning rate decay')
    parser.add_argument('--stop_at_epoch', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--labeled_ratio', type=float, default=0.8)
    parser.add_argument('--ratio_live', type=float, default=1.0)
    parser.add_argument('--ratio_spoof', type=float, default=1.0)
    parser.add_argument('--cross_start', type=float, default=0.0)
    parser.add_argument('--fully_labeled', type=int, default=0)
    parser.add_argument('--proj_layers', type=int, default=None, help="number of projector layers. In cifar experiment, this is set to 2")
    # optimization params
    parser.add_argument('--optimizer', type=str, default='sgd', help='sgd, lars(from lars paper), lars_simclr(used in simclr and byol), larc(used in swav)')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='learning rate will be linearly scaled during warm up period')
    parser.add_argument('--warmup_lr', type=float, default=0, help='Initial warmup learning rate')
    parser.add_argument('--base_lr', type=float, default=0.05)
    parser.add_argument('--final_lr', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)

    parser.add_argument('--eval_after_train', type=str, default=None)
    parser.add_argument('--head_tail_accuracy', action='store_true', help='the acc in first epoch will indicate whether collapse or not, the last epoch shows the final accuracy')
    # distributed params
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    # display params
    parser.add_argument('--print_freq', type=int, default=10, help='print_freq')

    # preprocess params
    parser.add_argument('--crop_scale', type=float, default=1.5)

    # epcr params
    parser.add_argument('--embedding_alpha', type=float, default=1.0)
    parser.add_argument('--prediction_alpha', type=float, default=0.1)

    # ocim dataset
    parser.add_argument('--ocim_sample_num', type=int, default=3)

    # wmca dataset 
    parser.add_argument('--wmca_grand_path', type=str, default=None)
    parser.add_argument('--wmca_img_path', type=str, default=None)
    parser.add_argument('--wmca_protocol_cmd', type=str, default=None)

    # ssdg configs
    parser.add_argument('--loss_weight_triplet', type=float, default=1.0)
    parser.add_argument('--loss_weight_adloss', type=float, default=0.1)

    # supplementary material configs
    parser.add_argument('--augment_choices', type=str, default=None)

    # environment
    parser.add_argument('--cudnn', type=int, default=1, help='use cudnn?')
    args = parser.parse_args()

    

    
    if args.debug:
        args.batch_size = 2 
        args.stop_at_epoch = 2
        args.num_epochs = 3 # train only one epoch
        args.num_workers = 0

    assert not None in [args.output_dir, args.data_dir]
    os.makedirs(args.output_dir, exist_ok=True)
    # assert args.stop_at_epoch <= args.num_epochs
    if args.stop_at_epoch is not None:
        if args.stop_at_epoch > args.num_epochs:
            raise Exception
    else:
        args.stop_at_epoch = args.num_epochs

    if args.use_default_hyperparameters:
        raise NotImplementedError
    return args
