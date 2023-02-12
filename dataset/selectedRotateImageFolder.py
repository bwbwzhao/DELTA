from ast import arg
import os
import copy
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image

class TensorDataset(Dataset):
    def __init__(self, data, targets):
        super(TensorDataset).__init__()
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return [img, target]

    def __len__(self):
        return len(self.targets)


def  dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels==y).flatten() 
           for y in range(n_classes)]
 
    client_idcs = []
    for _ in range(n_clients):
        class_tmp = []
        for _ in range(n_classes):
            class_tmp.append([])
        client_idcs.append(class_tmp)

    for c, (c_idcs, fracs) in enumerate(zip(class_idcs, label_distribution)):
        for i, idcs in enumerate(np.split(c_idcs, (np.cumsum(fracs)[:-1]*len(c_idcs)).astype(int))):
            client_idcs[i][c].append(idcs)

    client_idcs_final = []
    for client in client_idcs:
        random.shuffle(client)
        client_idcs_final.append(np.concatenate(client, axis=1)[0])

    return np.concatenate(client_idcs_final)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tr_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
									transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
									transforms.ToTensor(),
									normalize])
te_transforms = transforms.Compose([transforms.Resize(256),
									transforms.CenterCrop(224),
									transforms.ToTensor(),
									normalize])
te_transforms_imageC = transforms.Compose([transforms.CenterCrop(224),
									transforms.ToTensor(),
									normalize])

rotation_tr_transforms = tr_transforms
rotation_te_transforms = te_transforms

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
	                    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
	                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


class ImagePathFolder(datasets.ImageFolder):
	def __init__(self, traindir, train_transform):
		super(ImagePathFolder, self).__init__(traindir, train_transform)	

	def __getitem__(self, index):
		path, _ = self.imgs[index]
		img = self.loader(path)
		if self.transform is not None:
			img = self.transform(img)
		path, pa = os.path.split(path)
		path, pb = os.path.split(path)
		return img, 'val/%s/%s' %(pb, pa)


# =========================Rotate ImageFolder Preparations Start======================
# Assumes that tensor is (nchannels, height, width)
def tensor_rot_90(x):
	return x.flip(2).transpose(1, 2)

def tensor_rot_180(x):
	return x.flip(2).flip(1)

def tensor_rot_270(x):
	return x.transpose(1, 2).flip(2)

def rotate_single_with_label(img, label):
	if label == 1:
		img = tensor_rot_90(img)
	elif label == 2:
		img = tensor_rot_180(img)
	elif label == 3:
		img = tensor_rot_270(img)
	return img

def rotate_batch_with_labels(batch, labels):
	images = []
	for img, label in zip(batch, labels):
		img = rotate_single_with_label(img, label)
		images.append(img.unsqueeze(0))
	return torch.cat(images)

def rotate_batch(batch, label='rand'):
	if label == 'rand':
		labels = torch.randint(4, (len(batch),), dtype=torch.long)
	else:
		assert isinstance(label, int)
		labels = torch.zeros((len(batch),), dtype=torch.long) + label
	return rotate_batch_with_labels(batch, labels), labels


# =========================Rotate ImageFolder Preparations End======================

# The following ImageFolder supports sample a subset from the entire dataset by index/classes/sample number, at any time after the dataloader created. 
class SelectedRotateImageFolder(datasets.ImageFolder):
    def __init__(self, root, train_transform, original=True, rotation=True, rotation_transform=None):
        super(SelectedRotateImageFolder, self).__init__(root, train_transform)
        self.original = original
        self.rotation = rotation
        self.rotation_transform = rotation_transform

        self.original_samples = self.samples

    def __getitem__(self, index):
        # path, target = self.imgs[index]
        path, target = self.samples[index]
        img_input = self.loader(path)

        if self.transform is not None:
            img = self.transform(img_input)
        else:
            img = img_input

        results = []
        if self.original:
            results.append(img)
            results.append(target)
        if self.rotation:
            if self.rotation_transform is not None:
                img = self.rotation_transform(img_input)
            target_ssh = np.random.randint(0, 4, 1)[0]
            img_ssh = rotate_single_with_label(img, target_ssh)
            results.append(img_ssh)
            results.append(target_ssh)
        return results

    def switch_mode(self, original, rotation):
        self.original = original
        self.rotation = rotation

    def set_target_class_dataset(self, target_class_index, logger=None):
        self.target_class_index = target_class_index
        self.samples = [(path, idx) for (path, idx) in self.original_samples if idx in self.target_class_index]
        self.targets = [s[1] for s in self.samples]

    def set_dataset_size(self, subset_size):
        num_train = len(self.targets)
        indices = list(range(num_train))
        random.shuffle(indices)
        self.samples = [self.samples[i] for i in indices[:subset_size]]
        self.targets = [self.targets[i] for i in indices[:subset_size]]
        return len(self.targets)

    def set_specific_subset(self, indices):
        self.samples = [self.original_samples[i] for i in indices]
        self.targets = [s[1] for s in self.samples]


def reset_data_sampler(sampler, dset_length, dset):
    sampler.dataset = dset
    if dset_length % sampler.num_replicas != 0 and False:
        sampler.num_samples = math.ceil((dset_length - sampler.num_replicas) / sampler.num_replicas)
    else:
        sampler.num_samples = math.ceil(dset_length / sampler.num_replicas)
    sampler.total_size = sampler.num_samples * sampler.num_replicas


def prepare_train_dataset(args):
    print('Preparing data...')
    traindir = os.path.join(args.data, 'train')
    trset = SelectedRotateImageFolder(traindir, tr_transforms, original=True, rotation=args.rotation,
                                                        rotation_transform=rotation_tr_transforms)
    return trset


def prepare_train_dataloader(args, trset=None, sampler=None):
    if sampler is None:
        trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size, shuffle=True,
                                                        num_workers=args.workers, pin_memory=True)
        train_sampler = None
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trset)
        trloader = torch.utils.data.DataLoader(
            trset, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True) #sampler=None shuffle=True,
    return trloader, train_sampler


def prepare_test_data(args, idx=None, plot_colors=None, use_transforms=True):
    if args.corruption == 'original':
        te_transforms_local = te_transforms if use_transforms else None
    elif args.corruption in common_corruptions:
        te_transforms_local = te_transforms_imageC if use_transforms else None
    elif args.corruption=='rendition':
        te_transforms_local = te_transforms if use_transforms else None
    else:
        assert False, NotImplementedError
    if not hasattr(args, 'corruption') or args.corruption == 'original':
        print('Test on the original test set')
        validdir = os.path.join('/dockerdata/imagenet', 'val')
        teset = SelectedRotateImageFolder(validdir, te_transforms_local, original=False, rotation=False,
                                                    rotation_transform=rotation_te_transforms)
    elif args.corruption in common_corruptions:
        print('Test on %s level %d' %(args.corruption, args.level))
        validdir = os.path.join(args.data_corruption_dir, args.corruption, str(args.level))
        teset = SelectedRotateImageFolder(validdir, te_transforms_local, original=False, rotation=False,
                                                    rotation_transform=rotation_te_transforms)
    elif args.corruption=='rendition':
        assert 'imagenet-r' in args.data_corruption_dir
        validdir = os.path.join(args.data_corruption_dir)
        teset = SelectedRotateImageFolder(validdir, te_transforms_local, original=False, rotation=False,
                                                    rotation_transform=rotation_te_transforms)
        print('rendition...')
    else:
        raise Exception('Corruption not found!')
        
    if not hasattr(args, 'workers'):
        args.workers = 1

    if args.distri_type=='iid':
        if idx is None:
            idx = [i for i in range(50000)] if args.corruption!='rendition' else [i for i in range(30000)]
            random.shuffle(idx)
        teset.imgs = [teset.imgs[i] for i in idx]
        print(teset.imgs[:50])
        teset.original_samples = [teset.original_samples[i] for i in idx]
        teset.samples = [teset.samples[i] for i in idx]
        teset.targets = [teset.targets[i] for i in idx]
    elif args.distri_type=='noniid':
        if idx is None:
            idx = dirichlet_split_noniid(np.array(teset.targets), args.noniid_factor, 10)
        teset.imgs = [teset.imgs[i] for i in idx]
        teset.original_samples = [teset.original_samples[i] for i in idx]
        teset.samples = [teset.samples[i] for i in idx]
        teset.targets = [teset.targets[i] for i in idx]
    elif args.distri_type=='lt':
        if idx is None:
            prob_per_class = []
            for cls_idx in range(args.class_num):
                prob_per_class.append( args.imb_factor ** (cls_idx / (args.class_num - 1.0)) )
            prob_per_class = np.array(prob_per_class) / sum(prob_per_class)
            img_per_class = prob_per_class * len(teset.targets)
            idx = []
            y_test_np = np.array(teset.targets)
            for c, num in enumerate(img_per_class):
                all_of_c = np.where(y_test_np==c)[0]
                idx.append(np.random.choice(all_of_c, int(num)+1))
            idx = np.concatenate(idx)
            random.shuffle(idx)
        teset.imgs = [teset.imgs[i] for i in idx]
        teset.original_samples = [teset.original_samples[i] for i in idx]
        teset.samples = [teset.samples[i] for i in idx]
        teset.targets = [teset.targets[i] for i in idx]
    elif args.distri_type=='noniidlt':
        if idx is None:
            prob_per_class = []
            for cls_idx in range(args.class_num):
                prob_per_class.append( args.imb_factor ** (cls_idx / (args.class_num - 1.0)) )
            prob_per_class = np.array(prob_per_class) / sum(prob_per_class)
            img_per_class = prob_per_class * len(teset.targets)
            idx = []
            y_test_np = np.array(teset.targets)
            for c, num in enumerate(img_per_class):
                all_of_c = np.where(y_test_np==c)[0]
                idx.append(np.random.choice(all_of_c, int(num)+1))
            idx = np.concatenate(idx)
            idx2 = dirichlet_split_noniid(np.array([y_test_np[i] for i in idx]), args.noniid_factor, 10)
            idx = [idx[i] for i in idx2]
        teset.imgs = [teset.imgs[i] for i in idx]
        teset.original_samples = [teset.original_samples[i] for i in idx]
        teset.samples = [teset.samples[i] for i in idx]
        teset.targets = [teset.targets[i] for i in idx]

    teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size, shuffle=False, 
                                                    num_workers=args.workers, pin_memory=True)
    return teset, teloader, idx, plot_colors
