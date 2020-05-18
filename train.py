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
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader
from retinanet.utils import bbox_collate, MixedRandomSampler,count_parameters_in_MB
from retinanet import transform as transf
from retinanet import coco_eval
from retinanet import csv_eval
import retinanet


assert torch.__version__.split('.')[0] == '1'

import yaml
import json
print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', default="capsule", help='Dataset type, must be one of csv or coco.')
    #parser.add_argument('--coco_path', help='Path to COCO directory')
    #parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    #parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    #parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 0, 1, 18, 34, 50, 101, 152', type=int, default=1)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=1)

    parser = parser.parse_args(args)

    if parser.depth == 0:
        config = yaml.safe_load(open('./config_darts.yaml'))
    else:
        config = yaml.safe_load(open('./config.yaml'))

    dataset_means = json.load(open(config['dataset']['mean_file']))
    transform = Compose([
        transf.Augmentation(config['augmentation']),
        transf.ToFixedSize([config['inputsize']] * 2),  # inputsize x inputsizeの画像に変換
        transf.Normalize(dataset_means['mean'], dataset_means['std']),
        transf.HWCToCHW()
        ])
    dataset_all = MedicalBboxDataset(
        config['dataset']['annotation_file'],
        config['dataset']['image_root'])
    if 'class_integration' in config['dataset']:
        dataset_all = dataset_all.integrate_classes(
            config['dataset']['class_integration']['new'],
            config['dataset']['class_integration']['map'])
    
    train_all = dataset_all.split(config['dataset']['train'], config['dataset']['split_file'])
    train_all.set_transform(transform)
    train_normal = train_all.without_annotation()
    train_anomaly = train_all.with_annotation()
    n_fg_class = len(dataset_all.get_category_names()) 

    generator = torch.Generator()
    generator.manual_seed(0)
    sampler = MixedRandomSampler(
        [train_normal, train_anomaly],
        config['n_iteration'] * config['batchsize'],
        distributed=False,
        ratio=[config['negative_ratio'], 1],
        generator=generator)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 24, drop_last=False)

    dataloader_train = DataLoader(
        sampler.get_concatenated_dataset(),
        num_workers=8,
        batch_sampler=batch_sampler,
        collate_fn=bbox_collate)

    if parser.depth == 0:
        arch_all = dataset_all.split(config['dataset']['arch'], config['dataset']['split_file'])
        arch_all.set_transform(transform)
        arch_normal = arch_all.without_annotation()
        arch_anomaly = arch_all.with_annotation()

        generator = torch.Generator()
        generator.manual_seed(0)
        sampler = MixedRandomSampler(
            [arch_normal, arch_anomaly],
            config['n_iteration'] * config['batchsize'],
            distributed=False,
            ratio=[config['negative_ratio'], 1],
            generator=generator)
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 24, drop_last=False)

        dataloader_arch = DataLoader(
            sampler.get_concatenated_dataset(),
            num_workers=8,
            batch_sampler=batch_sampler,
            collate_fn=bbox_collate)
    else:
        dataloader_arch = dataloader_train


    transform = Compose([
        transf.ToFixedSize([config['inputsize']] * 2),  # inputsize x inputsizeの画像に変換
        transf.Normalize(dataset_means['mean'], dataset_means['std']),
        transf.HWCToCHW()
        ])


    dataset_val = dataset_all.split(config['dataset']['val'], config['dataset']['split_file'])
    dataset_val.set_transform(transform)

    arch = 'DARTS'
    # Create the model
    if parser.depth == 0:
        retinanet = model.darts(num_classes=n_fg_class)
        opt_a = optim.Adam(retinanet.arch_parameters(),lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-3)
    elif parser.depth == 1:
        genotype = eval(f"retinanet.genotypes.{arch}")
        retinanet = model.Final(num_classes=n_fg_class, genotype=genotype)
    elif parser.depth == 18:
        retinanet = model.resnet18(num_classes=n_fg_class, pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=n_fg_class, pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=n_fg_class, pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=n_fg_class, pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=n_fg_class, pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 0, 1, 18, 34, 50, 101, 152')

    print(f'param size = {count_parameters_in_MB(retinanet)}MB')
    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataloader_train)))
    
    for epoch_num in range(parser.epochs):
        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []
        iter_num = -1

        for data, data_arch in zip(dataloader_train, dataloader_arch):
            iter_num += 1
            optimizer.zero_grad()
            classification_loss, regression_loss = retinanet(data['img'].cuda().float(), annot = data['annot'])
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss
            if bool(loss == 0):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
            optimizer.step()

            if parser.depth == 0:
                opt_a.zero_grad()
                classification_loss_valid, regression_loss_valid = retinanet(data_arch['img'].cuda().float(), annot = data_arch['annot'])
                classification_loss_valid = classification_loss_valid.sum()
                regression_loss_valid = regression_loss_valid.sum()
                loss_valid = classification_loss_valid + regression_loss_valid
                loss_valid.backward()
                opt_a.step()

            loss_hist.append(float(loss))
            epoch_loss.append(float(loss))
            print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
            
            del classification_loss
            del regression_loss
            
        

        print('Evaluating dataset')
        coco_eval.evaluate_coco(dataset_val, retinanet)
        scheduler.step(np.mean(epoch_loss))

    retinanet.eval()
    if parser.depth != 0:
        torch.save(retinanet, f'/data/unagi0/masaoka/retinanet/model_final_{parser.depth}.pt')
    else:
        print(retinanet.module.genotype())


if __name__ == '__main__':
    main()
