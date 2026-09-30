import os
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import numpy as np

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import HorizontalFlip, Normalize, RandomRotate90, Resize, VerticalFlip

import arch as archs

import losses
from dataset import Dataset

from metrics import iou_score, indicators

from utils import AverageMeter, str2bool

from tensorboardX import SummaryWriter

import os


ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    parser.add_argument('--seed', default=1029, type=int,
                        help='random seed for model initialization and training')
    parser.add_argument('--dataseed', default=2981, type=int,
                        help='random seed for train/validation split')
    
    # model
    parser.add_argument('--arch', '-a', choices=ARCH_NAMES, default='CTKANLight')
    
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='busi', help='dataset name')      
    parser.add_argument('--data_dir', default='inputs', help='dataset dir')

    parser.add_argument('--output_dir', default='outputs', help='ouput dir')


    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'AdamW', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'AdamW', 'SGD']) +
                        ' (default: Adam)')

    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    parser.add_argument('--kan_lr', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--ctm_lr', default=1e-3, type=float,
                        metavar='LR', help='learning rate for CTKAN dynamics')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--no_kan', action='store_true')

    # Shared CTKAN settings for CTKANLight and CTKANMax.
    parser.add_argument('--CtmTicks', default=5, type=int,
                        help='T: number of internal CTKAN thought ticks')
    parser.add_argument('--CtmDhidden', default=64, type=int,
                        help='hidden width inside each private NLM')
    parser.add_argument('--CtmDropout', default=0.2, type=float,
                        help='dropout probability inside CTM synapse model')
    parser.add_argument('--CtmDaction', default=1024, type=int,
                        help='action synchronization representation width for random pairing')
    parser.add_argument('--CtmDout', default=1024, type=int,
                        help='output synchronization representation width for random pairing')
    parser.add_argument('--CtmNself', default=32, type=int,
                        help='number of self-pairs used in random synchronization sampling')
    parser.add_argument('--CtmMemory', default=5, type=int,
                        help='L: exponential-memory time scale')
    parser.add_argument('--CtmScaleInit', default=5e-2, type=float,
                        help='initial residual scale for CTM gate branch')
    parser.add_argument('--CtmActivityStd', default=0.5, type=float,
                        help='weight of token-wise spline-activity standard deviation')
    parser.add_argument('--CtmPriorScale', default=0.05, type=float,
                        help='shared-state prior scale in input-conditioned initialization')
    parser.add_argument('--CtmActivityEps', default=1e-6, type=float,
                        help='minimum RMS used to normalize spline activity')
    parser.add_argument('--CtmSyncEps', default=1e-4, type=float,
                        help='minimum RMS used to normalize synchronization products')
    parser.add_argument('--CtmTerminalReadout', default=True, type=str2bool,
                        help='use the independent terminal pair-readout MLP')
    parser.add_argument('--CtmPriorMode', default='factorized',
                        choices=['full', 'factorized', 'none'],
                        help='parameterization of the sample-independent prior')
    parser.add_argument('--CtmLinearLayout', default='first',
                        choices=['all', 'first', 'middle', 'last', 'first_last'],
                        help='which spline-linear units run recurrent CTKAN')
    parser.add_argument('--CtmGridSize', default=5, type=int,
                        help='number of spline grid intervals')
    parser.add_argument('--CtmSplineOrder', default=3, type=int,
                        help='B-spline polynomial order')

    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume training from checkpoint')



    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)

            iou, dice, _ = iou_score(outputs[-1], target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(outputs[-1], target)
            
        else:
            output = model(input)
            loss = criterion(output, target)
            iou, dice, _ = iou_score(output, target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice, _ = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice, _ = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()


    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    config = vars(parse_args())
    seed_torch(config['seed'])

    model_name = config.get('name') if config.get('name') else config['arch']
    output_dir = config.get('output_dir')
    save_dir = os.path.join(output_dir, config['dataset'], model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Clean old TensorBoard event files for this run directory.
    for f in glob(os.path.join(save_dir, 'events.out.tfevents.*')):
        os.remove(f)

    my_writer = SummaryWriter(save_dir)

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_wDS' % config['arch']
        else:
            config['name'] = '%s_woDS' % config['arch']

    print(f"Model: {config['arch']}")
    print(f"Dataset: {config['dataset']}")
    print("Starting training...")

    with open(os.path.join(save_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model
    model = getattr(archs, config['arch'])(
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        img_size=config['input_h'],
        embed_dims=config['input_list'],
        no_kan=config['no_kan'],
        CtmTicks=config['CtmTicks'],
        CtmDhidden=config['CtmDhidden'],
        CtmDropout=config['CtmDropout'],
        CtmDaction=config['CtmDaction'],
        CtmDout=config['CtmDout'],
        CtmNself=config['CtmNself'],
        CtmMemory=config['CtmMemory'],
        CtmScaleInit=config['CtmScaleInit'],
        CtmActivityStd=config['CtmActivityStd'],
        CtmPriorScale=config['CtmPriorScale'],
        CtmActivityEps=config['CtmActivityEps'],
        CtmSyncEps=config['CtmSyncEps'],
        CtmTerminalReadout=config['CtmTerminalReadout'],
        CtmPriorMode=config['CtmPriorMode'],
        CtmLinearLayout=config['CtmLinearLayout'],
        CtmGridSize=config['CtmGridSize'],
        CtmSplineOrder=config['CtmSplineOrder'],
    )

    model = model.cuda()


    param_groups = []

    ctm_dynamics_markers = (
        'z_init',
        'input_gain',
        'state_gain',
        'output_scale',
        'action_to_factors',
        'output_to_factors',
        'recurrent_gate',
    )

    for name, param in model.named_parameters():
        # print(name, "=>", param.shape)
        lower_name = name.lower()
        if any(marker in lower_name for marker in ctm_dynamics_markers):
            # Coupled L2 regularization in Adam drives scalar CTKAN gains to
            # zero very quickly. Keep the dynamics branch in a decay-free group.
            param_groups.append({
                'params': param,
                'lr': config['ctm_lr'],
                'weight_decay': 0.0,
            })
        elif 'layer' in lower_name and 'fc' in lower_name:
            param_groups.append({'params': param, 'lr': config['kan_lr'], 'weight_decay': config['kan_weight_decay']}) 
        else:
            param_groups.append({'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay']})  
    

    
    # st()
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)


    elif config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(param_groups)


    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(param_groups, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    start_epoch = 0
    best_iou = 0
    best_dice = 0

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])

    # resume from checkpoint
    if config['resume']:
        ckpt_path = os.path.join(save_dir, 'checkpoint.pth')
        log_path = os.path.join(save_dir, 'log.csv')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_iou = checkpoint.get('best_iou', 0)
            best_dice = checkpoint.get('best_dice', 0)
            print(f"=> resumed from checkpoint at epoch {checkpoint.get('epoch', 0)}")
            # load existing log
            if os.path.exists(log_path):
                log_df = pd.read_csv(log_path)
                log['epoch'] = list(log_df['epoch'].values[:start_epoch])
                log['lr'] = list(log_df['lr'].values[:start_epoch])
                log['loss'] = list(log_df['loss'].values[:start_epoch])
                log['iou'] = list(log_df['iou'].values[:start_epoch])
                log['val_loss'] = list(log_df['val_loss'].values[:start_epoch])
                log['val_iou'] = list(log_df['val_iou'].values[:start_epoch])
                log['val_dice'] = list(log_df['val_dice'].values[:start_epoch])
        else:
            raise FileNotFoundError(f"=> resume requested, but no checkpoint found at {ckpt_path}")

    dataset_name = config['dataset']
    img_ext = '.png'

    if dataset_name == 'busi':
        mask_ext = '_mask.png'
    elif dataset_name == 'glas':
        mask_ext = '.png'
    elif dataset_name == 'heus':
        mask_ext = '.png'
    elif dataset_name == 'cvc':
        mask_ext = '.png'
    elif dataset_name in ('brisc', 'lits'):
        mask_ext = '_mask.png'
    else:
        mask_ext = '.png'

    # Data loading code
    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    mask_dir_name = 'masks'

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])

    train_transform = Compose([
        RandomRotate90(),
        HorizontalFlip(),
        VerticalFlip(),
        Resize(config['input_h'], config['input_w']),
        Normalize(),
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], mask_dir_name),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_dir'] ,config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], mask_dir_name),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    trigger = 0
    for epoch in range(start_epoch, config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv(os.path.join(save_dir, 'log.csv'), index=False)

        my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)
        my_writer.add_scalar('train/iou', train_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch)
        my_writer.add_scalar('val/iou', val_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/dice', val_log['dice'], global_step=epoch)

        my_writer.add_scalar('val/best_iou_value', best_iou, global_step=epoch)
        my_writer.add_scalar('val/best_dice_value', best_dice, global_step=epoch)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
            best_iou = val_log['iou']
            best_dice = val_log['dice']
            print("=> saved best model")
            print('IoU: %.4f' % best_iou)
            print('Dice: %.4f' % best_dice)
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        # save checkpoint for resume
        checkpoint = {
            'epoch': int(epoch),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_iou': float(best_iou),
            'best_dice': float(best_dice),
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))

        torch.cuda.empty_cache()
    
if __name__ == '__main__':
    main()
