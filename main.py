import os
from pyexpat import model
import time
import argparse
import random
from utils.utils import get_logger
from utils.cli_utils import *
from dataset.selectedRotateImageFolder import prepare_test_data
import torch    
import torch.nn.functional as F
import numpy as np
import models.Res as Resnet
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pylab as plt
import yaml
from easydict import EasyDict
import torchvision


def validate(val_loader, model, args):
    class_num = np.zeros(args.class_num)
    class_correct = np.zeros(args.class_num)
    pred_counts = np.zeros(args.class_num)

    for iter, (images, target) in enumerate(val_loader):
        if (iter+1)%10==0:
            acc = (class_correct / class_num)[class_num!=0]
            print(f'Iter {iter} / {len(val_loader)}, {acc.mean() * 100}, {class_correct.sum() / class_num.sum() * 100.}....')
        images, target = images.cuda(), target.cuda()
        output = model(images, target)

        with torch.no_grad():
            preds = output.argmax(dim=1)
            for i, t in enumerate(target):
                class_num[t.item()] += 1
                class_correct[t.item()] += (preds[i]==t)
                pred_counts[preds[i]] += 1

    acc = (class_correct / class_num)[class_num!=0]
    acc = acc.mean() * 100.
    return acc, class_correct.sum() / class_num.sum() * 100.


def get_adapt_model(args, subnet):
    if args.algorithm == 'source':
        from algorithms.source import Source
        adapt_model = Source(args, subnet)
    elif args.algorithm == 'norm':
        from algorithms.norm import Norm
        adapt_model = Norm(args, subnet)
    elif args.algorithm == 'delta':
        from algorithms.delta import DELTA
        adapt_model = DELTA(args, subnet)
    else:
        raise NotImplementedError
    return adapt_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet-C Testing')
    parser.add_argument('--cfg', default='./configs/imagenet/imagenetc_norm.yaml', type=str)
    parser.add_argument('--seed', default=2020, type=int)
    parser.add_argument('--distri_type', default='iid', type=str)
    tmp_args = parser.parse_known_args()[0]
    args = yaml.safe_load(open(tmp_args.cfg))
    if tmp_args.seed is not None:
        args['seed'] = tmp_args.seed
        args['prefix'] = args['prefix']+str(tmp_args.seed)
    assert tmp_args.distri_type is not None
    types = tmp_args.distri_type.split('_')
    args['distri_type'] = types[0]
    if args['distri_type']=='iid':
        pass
    elif args['distri_type']=='noniid':
        args['noniid_factor'] = float(types[1])
    elif args['distri_type']=='lt':
        args['imb_factor'] = float(types[1])
    elif args['distri_type']=='noniidlt':
        args['noniid_factor'] = float(types[1])
        args['imb_factor'] = float(types[2])
    args = EasyDict(args)

    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        # random.seed(2020)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # prepare model
    subnet = Resnet.__dict__[args.arch](pretrained=True).cuda()
    adapt_model = get_adapt_model(args, subnet)

    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                             'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                             'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

    # outs
    if args.distri_type=='iid':
        args.output = args.output + '/' + args.distri_type + '/'
    elif args.distri_type=='noniid':
        args.output = args.output + '/' + args.distri_type+str(args.noniid_factor) + '/'
    elif args.distri_type=='lt':
        args.output = args.output + '/' + args.distri_type+str(args.imb_factor) + '/'
    elif args.distri_type=='noniidlt':
        args.output = args.output + '/' + args.distri_type+str(args.noniid_factor)+'_'+str(args.imb_factor) + '/'

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    args.prefix = args.algorithm + '_' + args.prefix + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    writer = SummaryWriter(args.output + args.prefix)
    logger = get_logger(name="project", output_directory = args.output + args.prefix, log_name="-log.txt", debug=False)
    logger.info(args)

    top1_list = []    
    idx = None
    plot_colors = None
    for type_id, corrupt in enumerate(common_corruptions):
        # prepare data
        args.corruption = corrupt
        logger.info(args.corruption)

        val_dataset, val_loader, idx, plot_colors = prepare_test_data(args, idx=idx, plot_colors=plot_colors)
        logger.info(idx[:50])

        if 'imagenet' in args.dataset:
            val_dataset.switch_mode(True, False)

        if args.exp_type == 'each_shift_reset':
            adapt_model.reset()
        else:
            NotImplementedError

        top1, top1_ = validate(val_loader, adapt_model, args)

        top1_list.append(top1)
        logger.info(f"Under shift type {args.corruption} After {args.algorithm} \
            Top1: {top1:.5f} Err: {100. - top1:.5f} ||| \
            Mean Top1: {np.stack(top1_list).mean():.5f} Mean Err: {100. - np.stack(top1_list).mean():.5f}"
        )
        logger.info(f"Under shift type {args.corruption} After {args.algorithm} \
            Top1: {top1_:.5f} Err: {100. - top1_:.5f}"
        )
        writer.add_scalar('top1', top1, type_id)
        writer.add_scalar('err', 100. - top1, type_id)
        writer.add_scalar('mean top1', np.stack(top1_list).mean(), type_id)
        writer.add_scalar('mean err', 100. - np.stack(top1_list).mean(), type_id)
        # break
    writer.close()
