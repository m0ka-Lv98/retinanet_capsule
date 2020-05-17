import argparse
import collections
from tqdm import tqdm
import numpy as np
from retinanet.dataset import MedicalBboxDataset
import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import Compose
from retinanet import model

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import DataLoader
from retinanet.utils import bbox_collate, MixedRandomSampler
from retinanet import transform as transf
from retinanet import coco_eval
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

import yaml
import json


import torchvision
import time
import os
import copy
import pdb

import sys
import cv2

from torch.utils.data import Dataset
from torchvision import datasets, models

config = yaml.safe_load(open('./config.yaml'))
dataset_means = json.load(open(config['dataset']['mean_file']))
dataset_all = MedicalBboxDataset(
    config['dataset']['annotation_file'],
    config['dataset']['image_root'])
if 'class_integration' in config['dataset']:
    dataset_all = dataset_all.integrate_classes(
            config['dataset']['class_integration']['new'],
            config['dataset']['class_integration']['map'])
transform = Compose([
        transf.ToFixedSize([config['inputsize']] * 2),  # inputsize x inputsizeの画像に変換
        transf.Normalize(dataset_means['mean'], dataset_means['std']),
        transf.HWCToCHW()
        ])
dataset_val = dataset_all.split(config['dataset']['val'], config['dataset']['split_file'])
dataset_val.set_transform(transform)

sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=bbox_collate, batch_sampler=sampler_val)

retinanet = torch.load("model_final.pt")
retinanet = retinanet.cuda()
retinanet.eval()
unnormalize = UnNormalizer(dataset_means['mean'], dataset_means['std'])
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    
for idx, data in enumerate(dataloader_val):
    with torch.no_grad():
        st = time.time()
        if torch.cuda.is_available():
            scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
        else:
            scores, classification, transformed_anchors = retinanet(data['img'].float())
        
        print('Elapsed time: {}'.format(time.time()-st))
        idxs = np.where(scores.cpu()>0.5)
        img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()
        img[img<0] = 0
        img[img>255] = 255
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            label_name = dataset_val.labels[int(classification[idxs[0][j]])]
            draw_caption(img, (x1, y1, x2, y2), label_name)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            print(label_name)
        cv2.imshow('img', img)
        cv2.waitKey(0)

