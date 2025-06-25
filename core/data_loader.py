"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class DefaultDataset(ImageFolder):
    def __init__(self, root, mask_root=None, transform=None, transform_mask=None, max_per_class=None):
        super().__init__(root, transform=None)
        if max_per_class is not None:
            print('Limiting number of images per class...')
            samples_by_class = {}
            for path, target in self.samples:
                if target not in samples_by_class:
                    samples_by_class[target] = []
                samples_by_class[target].append((path, target))

            new_samples = []
            for target in sorted(samples_by_class.keys()):
                samples = samples_by_class[target]
                if len(samples) > max_per_class:
                    samples = random.sample(samples, max_per_class)
                new_samples.extend(samples)
            self.samples = new_samples
            self.targets = [s[1] for s in self.samples]

        self.transform = transform
        self.transform_mask = transform_mask
        self.mask_root = mask_root
        self.loader = lambda path: Image.open(path).convert('RGB')

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        seed = np.random.randint(2147483647)
        if self.transform is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            sample = self.transform(sample)

        if self.mask_root is not None:
            rel_path = Path(path).relative_to(self.root)
            rel_path_png = rel_path.with_suffix('.png')
            mask_path = Path(self.mask_root) / rel_path_png
            mask = Image.open(mask_path).convert('L')

            if self.transform_mask is not None:
                random.seed(seed)
                torch.manual_seed(seed)
                mask = self.transform_mask(mask)
            
            return sample, target, mask
        else:
            return sample, target

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None, max_per_class=None):
        self.samples, self.targets = self._make_dataset(root, max_per_class)
        self.transform = transform

    def _make_dataset(self, root, max_per_class=None):
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            if max_per_class is not None and len(cls_fnames) > max_per_class:
                cls_fnames = random.sample(cls_fnames, max_per_class)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label

    def __len__(self):
        return len(self.targets)


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(img_root, mask_root=None, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4, max_per_class=2200):
    print(f'Preparing DataLoader to fetch {which} images during the training phase...')

    crop = transforms.RandomResizedCrop(img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(lambda x: crop(x) if random.random() < prob else x)

    transform_img = transforms.Compose([
        rand_crop,
        transforms.Resize([img_size, img_size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    transform_mask = transforms.Compose([
        rand_crop,
        transforms.Resize([img_size, img_size], interpolation=Image.NEAREST),  # use NEAREST for masks
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    if which == 'source':
        dataset = DefaultDataset(img_root, mask_root, transform=transform_img, transform_mask=transform_mask, max_per_class=max_per_class)
    elif which == 'reference':
        # Unpaired reference images without segmentation masks
        dataset = ReferenceDataset(img_root, transform=transform_img, max_per_class=max_per_class)
    else:
        raise NotImplementedError(f"Unsupported loader type: {which}")

    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)



def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = DefaultDataset(root, transform=transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = ImageFolder(root, transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode
        self.iter = iter(self.loader)
        if self.loader_ref is not None:
            self.iter_ref = iter(self.loader_ref)

    def _fetch_inputs(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            return next(self.iter)

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except StopIteration:
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        if self.mode == 'train':
            x, y, mask = self._fetch_inputs()
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2,
                           seg_gt=mask)
        elif self.mode == 'val':
            x, y = self._fetch_inputs()
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            x, y = self._fetch_inputs()
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})