import logging
import math
import random

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import torch
from .randaugment import RandAugmentMC
from skimage import filters
logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std), 
        args=args, isunlabels=True)

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args, root):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std),
        args=args, isunlabels=True)

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    
    labeled_idx = np.array(labeled_idx)
    
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(), # 水平反转
            transforms.RandomCrop(size=32, # 裁剪
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            # RandAugmentMC(n=2, m=10, cutout=False)])
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, img, simimg=None):

        # 需要进行mix时
        if not simimg is None: # x
            weak = self.weak(img)
            # 随机生成替换后的尺寸
            w = random.randint(1, img.width // 4) # 最大1/3
            h = random.randint(1, img.height // 4)
            x = random.randint(0, img.width - w)
            y = random.randint(0, img.height - h)
            img.paste(simimg.crop((x, y, x+w, y+h)), (x, y))
            # img.paste(simimg.resize((w, h)), (x, y))
            strong = self.strong(img)
        else:
            weak = self.weak(img)
            strong = self.strong(img)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, args=None, isunlabels=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        
        self.sim_ratio = 0
        
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        
        self.isunlabels = isunlabels
        if self.isunlabels:
            self.id_loss = np.zeros(len(self.data))  # 记录当前样本损失
            self.pseLabel = np.ones_like(self.targets) * -1   # 记录预测类别
            self.simMatrix = np.ones(args.num_classes)
            self.easy_mask = np.zeros(len(self.data)) # 是否已经容易区分

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        # 获取相似的图像融合后，在mix,无标签数据扰动
        sim_img = None
        if self.isunlabels:
            sim_img = self.get_sim_img(index)
            img = self.transform(img, sim_img)
            return img, target, index

        # 图像增强
        if self.transform is not None:
            img = self.transform(img)
        
        # 标签增强
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_sim_img(self,index):
        if self.easy_mask[index] == False: return None  # 不需要合并
        current_cls = self.pseLabel[index]  # 当前类别
        if current_cls==-1: return None  # 不需要合并
        sim_cls = self.simMatrix[current_cls]  # 相似类别
        
        # 获取相似的图像
        mask = self.pseLabel==sim_cls
        mask = np.where(mask)[0] # 找到是True
        if mask.shape[0] < 1: return None  # 没有相似类别的图像
        random_index = np.random.choice(mask)
        
        return Image.fromarray(self.data[random_index])

    
    # 更新当前的损失
    @torch.no_grad()
    def updata_data(self, idx, current_loss, current_cls, current_sim, epoch):
        # 更新损失
        self.id_loss[idx] = 0.001*self.id_loss[idx] + 0.999*current_loss.detach().cpu().numpy()
        # 更新标签
        self.pseLabel[idx] = current_cls.cpu().numpy()
        # 更新相似矩阵
        self.simMatrix = current_sim.cpu().numpy()
        
        # 计算阈值
        if epoch>=5:  # 第五个epochs之后在做混合
            g = filters.threshold_otsu(self.id_loss) # 大津法获取阈值
            self.easy_mask[self.id_loss<g]=True  # 容易区分的混合
            self.easy_mask[self.id_loss>g]=False # 不易区分的不混合
            self.sim_ratio = self.easy_mask.sum() / self.easy_mask.shape[0]
    
    def get_easy_ratio(self):
        return self.easy_mask.sum(-1) / self.easy_mask.shape[0]
    
class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, args=None, isunlabels=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.isunlabels = isunlabels
        if self.isunlabels:
            self.id_loss = np.zeros(len(self.data))  # 记录当前样本损失
            self.pseLabel = np.ones_like(self.targets) * -1   # 记录预测类别
            self.simMatrix = np.ones(args.num_classes)
            self.easy_mask = np.zeros(len(self.data)) # 是否已经容易区分
        self.sim_ratio = 0
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        # 获取相似的图像融合后，在mix,无标签数据扰动
        sim_img = None
        if self.isunlabels:
            sim_img = self.get_sim_img(index)
            img = self.transform(img, sim_img)
            return img, target, index


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
    def get_sim_img(self,index):
        if self.easy_mask[index] == False: return None  # 不需要合并
        current_cls = self.pseLabel[index]  # 当前类别
        if current_cls==-1: return None  # 不需要合并
        sim_cls = self.simMatrix[current_cls]  # 相似类别
        
        # 获取相似的图像
        mask = self.pseLabel==sim_cls
        mask = np.where(mask)[0] # 找到是True
        if mask.shape[0] < 1: return None  # 没有相似类别的图像
        random_index = np.random.choice(mask)
        
        return Image.fromarray(self.data[random_index])

    # 更新当前的损失
    @torch.no_grad()
    def updata_data(self, idx, current_loss, current_cls, current_sim, epoch):
        # 更新损失
        self.id_loss[idx] = 0.001*self.id_loss[idx] + 0.999*current_loss.detach().cpu().numpy()
        # 更新标签
        self.pseLabel[idx] = current_cls.cpu().numpy()
        # 更新相似矩阵
        self.simMatrix = current_sim.cpu().numpy()
        # 计算阈值
        if epoch>=5:  # 第五个epochs之后在做混合
            g = filters.threshold_otsu(self.id_loss) # 大津法获取阈值
            self.easy_mask[self.id_loss<g]=True  # 容易区分的混合
            self.easy_mask[self.id_loss>g]=False # 不易区分的不混合
            self.sim_ratio = self.easy_mask.sum() / self.easy_mask.shape[0]

     
    def get_easy_ratio(self):
        return self.easy_mask.sum(-1) / self.easy_mask.shape[0]
 
    
DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100}
