"""
Reference: https://github.com/cleinc/bts (From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation)
"""

import pandas as pd
import os
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import random
import scipy.ndimage as ndimage
# from .transforms import *

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])

class DataLoadPreprocess(Dataset):
    def __init__(self, csv_file, config, mode, transform=None, eval=False):
        self.paths = pd.read_csv(csv_file, header=None,
                                     names=['image', 'depth'])
        self.config = config
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor

        self.dataset_path = os.path.dirname(os.path.dirname(csv_file))

        self.input_height = self.config.DATA.TRAIN_CROP_SIZE[1]
        self.input_width = self.config.DATA.TRAIN_CROP_SIZE[0]

        self.do_random_rotate=True
        self.degree = 2.5

    def __getitem__(self, idx):

        image_path = self.dataset_path + "/"+ self.paths['image'][idx]
        depth_path = self.dataset_path + "/"+ self.paths['depth'][idx]

        image = Image.open(image_path)
        depth_gt = Image.open(depth_path)

        depth_gt = depth_gt.crop((41, 45, 601, 471))
        image = image.crop((41, 45, 601, 471))

        if self.mode == 'train':
            depth_completed_path = depth_path #self.dataset_path + '/' + self.paths['depth'][idx].replace("train","train_completed")
            depth_completed = Image.open(depth_completed_path)
            depth_completed = depth_completed.crop((41, 45, 601, 471))

        if self.mode == 'train' and self.do_random_rotate is True:
            random_angle = (random.random() - 0.5) * 2 * self.degree
            image = self.rotate_image(image, random_angle)
            depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
            depth_completed = self.rotate_image(depth_completed, random_angle, flag=Image.NEAREST)

        image = np.asarray(image, dtype=np.float32) / 255.0
        depth_gt = np.asarray(depth_gt, dtype=np.float32)
        depth_gt = np.expand_dims(depth_gt, axis=2)

        if self.mode == 'train':
            depth_gt = depth_gt * 1000.0
            depth_gt = depth_gt / 255.0
        else:
            depth_gt = depth_gt / 1000.0

        if self.mode == 'train':
            depth_completed = np.asarray(depth_completed, dtype=np.float32)
            depth_completed = np.expand_dims(depth_completed, axis=2)
            depth_completed = depth_completed * 1000.0
            depth_completed = depth_completed / 255.0

            image, depth_gt, depth_completed = self.random_crop(image, depth_gt, depth_completed, self.input_height, self.input_width)
            image, depth_gt, depth_completed = self.train_preprocess(image, depth_completed, depth_gt)

            depth_gt = np.clip(depth_gt, 10.0, 1000.0)
            depth_gt = 1000. / depth_gt

            depth_completed = np.clip(depth_completed, 10.0, 1000.0)
            depth_completed = 1000. / depth_completed

            sample = {'image': image, 'depth': depth_gt, 'depth_completed': depth_completed}
        else:
            sample = {'image': image, 'depth': depth_gt}

        if self.transform:
            sample = self.transform(sample)

        if self.mode != 'train':
            #print(os.path.basename(self.paths['image'][idx]))
            sample['path'] = os.path.basename(self.paths['image'][idx]) #) #.replace("_colors", "")

        # sample['path'] = os.path.basename(self.paths['image'][idx])  # ) #.replace("_colors", "")

        # print("min image: " + str(torch.min(sample['image']).item()) + " max image: " + str(torch.max(sample['image']).item()))
        # print("min depth: " + str(torch.min(sample['depth']).item()) + " max depth: " + str(torch.max(sample['depth']).item()))

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, depth_completed, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        assert img.shape[0] == depth_completed.shape[0]
        assert img.shape[1] == depth_completed.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        depth_completed = depth_completed[y:y + height, x:x + width, :]
        return img, depth, depth_completed

    def train_preprocess(self, image, depth_gt, depth_completed):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
            depth_completed = (depth_completed[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt, depth_completed

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        brightness = random.uniform(0.75, 1.25)

        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        #return 500
        return len(self.paths)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image = sample['image']
        image = self.to_tensor(image)
        image = self.normalize(image)

        depth = sample['depth']
        depth = self.to_tensor(depth)

        if 'depth_completed' in sample.keys():
            depth_completed = sample['depth_completed']
            depth_completed = self.to_tensor(depth_completed)
            return {'image': image, 'depth': depth, 'depth_completed':depth_completed}
        else:
            return {'image': image, 'depth': depth}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class Nyu2DataModule(LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.train_csv_path = os.path.join(config.DATASET.PATH, config.DATASET.TYPE, 'data' , 'nyu2_train.csv')
        self.test_csv_path = os.path.join(config.DATASET.PATH, config.DATASET.TYPE, 'data' , 'nyu2_test.csv')

        mode = "train"
        self.training_samples = DataLoadPreprocess(self.train_csv_path, self.config, mode, transform=preprocessing_transforms(mode))

        self.train_loader = DataLoader(self.training_samples, self.config.SOLVER.BATCHSIZE,
                               shuffle=True,
                               num_workers=self.config.SOLVER.NUM_WORKERS,
                               pin_memory=True,
                               sampler=None)

        mode = "test"
        self.testing_samples = DataLoadPreprocess(self.test_csv_path, self.config, mode, transform=preprocessing_transforms(mode))


    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self, eval=False, shuffle=False):

        self.test_loader = DataLoader(self.testing_samples, 1,
                               shuffle = shuffle,
                               num_workers=1,
                               pin_memory=True,
                               sampler=None)

        return self.test_loader

class Nyu2DataModuleTest(LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.test_csv_path = os.path.join(config.DATASET.PATH, config.DATASET.TYPE, 'test.csv')

        mode = "test"
        self.testing_samples = DataLoadPreprocess(self.test_csv_path, self.config, mode, transform=preprocessing_transforms(mode))


    def val_dataloader(self, eval=False, shuffle=False):

        self.test_loader = DataLoader(self.testing_samples, 1,
                               shuffle = shuffle,
                               num_workers=1,
                               pin_memory=True,
                               sampler=None)

        return self.test_loader