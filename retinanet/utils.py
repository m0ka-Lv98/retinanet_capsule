import torch
import torch.nn as nn
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            if torch.cuda.is_available():
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
            else:
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))

        else:
            self.mean = mean
        if std is None:
            if torch.cuda.is_available():
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
            else:
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std

    def forward(self, boxes, deltas):

        widths  = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x   = boxes[:, :, 0] + 0.5 * widths
        ctr_y   = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)
      
        return boxes

def bbox_collate(batch):
    collated = {}
    
    for key in batch[0]:
        collated[key] = [torch.from_numpy(b[key]) for b in batch]
    
    collated['img'] = torch.stack(collated['img'], dim=0).to(torch.float)
    
    return collated


class InfiniteSampler:
    '''
    与えられたLength内に収まる数値を返すIterator
    '''
    def __init__(self, length, random=True, generator=None):
        self.length = length
        self.random = random
        if random:
            self.generator = torch.Generator() if generator is None else generator
        self.stock = []
        
    def __iter__(self):
        while True:
            yield self.get(1)[0]
    
    def get(self, n):
        while len(self.stock) < n:
            self.extend_stock()
        
        indices = self.stock[:n]
        self.stock = self.stock[n:]
        
        return indices
        
    def extend_stock(self):
        if self.random:
            self.stock += torch.randperm(self.length, generator=self.generator).numpy().tolist()
        else:
            self.stock += list(range(self.length))


class MixedRandomSampler(torch.utils.data.sampler.Sampler):
    '''
    複数のデータセットを一定の比で混ぜながら、指定した長さだけIterationするSampler
    '''
    def __init__(self, datasets, length, distributed=False, ratio=None, generator=None):
        self.catdataset = torch.utils.data.ConcatDataset(datasets)
        self.length = length
        
        if distributed:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            assert generator is not None
            assert length % self.world_size == 0
        else:
            self.rank = 0
            self.world_size = 1
        
        self.generator = torch.Generator() if generator is None else generator
        
        self.dataset_lengths = [len(dataset) for dataset in datasets]
        if ratio is None:
            self.ratio = torch.tensor(self.dataset_lengths, dtype=torch.float)
        else:
            self.ratio = torch.tensor(ratio, dtype=torch.float)
            
        self.samplers = [InfiniteSampler(l, generator=self.generator) for l in self.dataset_lengths]
    
    def __iter__(self):
        start_with = torch.cumsum(torch.tensor([0] + self.dataset_lengths), dim=0)
        selected = self.random_choice(self.ratio, self.length)
        
        indices = torch.empty(self.length, dtype=torch.int)
        
        for i in range(len(self.ratio)):
            mask = selected == i
            n_selected = mask.sum().item()
            indices[mask] = torch.tensor(self.samplers[i].get(n_selected), dtype=torch.int) + start_with[i]
        
        indices = indices.numpy().tolist()[self.rank::self.world_size]
        
        return iter(indices)
    
    def __len__(self):
        return int(self.length / self.world_size)
    
    def get_concatenated_dataset(self):
        return self.catdataset
    
    def random_choice(self, p, size):
        random = torch.rand(size, generator=self.generator)
        bins = torch.cumsum(p / p.sum(), dim=0)
        choice = torch.zeros(size, dtype=torch.int)

        for i in range(len(p) - 1):
            choice[random > bins[i]] = i + 1

        return choice