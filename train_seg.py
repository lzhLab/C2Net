import os
import sys
import argparse
import logging
# import time
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader

from models import *
from criterions import *
from datasets.dataset_livs import *
from trainer import *


def parse_args():
    parser = argparse.ArgumentParser()
    
    # path
    working_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(working_dir, 'results')
    datesets_dir = os.path.join(working_dir, 'datasets')

    parser.add_argument('--data-dir', type=str, metavar='PATH', default=datesets_dir)
    parser.add_argument('--dataset-name', type=str, default='data/LiVS/label')
    # parser.add_argument('--dataset-name', type=str, default='data/MSD/image')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=os.path.join(results_dir, 'logs'))
    parser.add_argument('--weights-dir', type=str, metavar='PATH',
                        default=os.path.join(results_dir, 'weights'))
    parser.add_argument('--predicts-dir', type=str, metavar='PATH',
                        default=os.path.join(results_dir, 'predicts'))
    parser.add_argument('--plots-dir', type=str, metavar='PATH',
                        default=os.path.join(results_dir, 'plots'))

    parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')

    # model
    parser.add_argument('--expand_size', type=int, default=1)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('-a', '--arch', type=str, default='RNN_Model',
                        choices=['Unet', 'RNN_Model'])
    parser.add_argument('--output-size', type=int, default=1)
    # parser.add_argument('--dropout', type=float, default=0.2)

    # criterion
    parser.add_argument('-c', '--criterion', type=str, default='db_criterion',
                        choices=['MSELoss', 'BCEWithLogitsLoss', 'db_criterion'])

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--lr-patience', type=int, default=40)
    parser.add_argument('--accumulate-step', type=int, default=1)

    # training configs
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-j', '--workers', type=int, default=8)

    args = parser.parse_args()
    return args


def getModel(args, device):
    modelDict = {
        'Unet': Unet,
        'RNN_Model': RNN_Model
    }

    model = modelDict[args.arch](args.expand_size * 2 + 1, args.output_size)
    model = nn.DataParallel(model).to(device)

    return model

def getCriterion(args):
    if args.criterion == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif args.criterion == 'MSELoss':
        criterion = nn.MSELoss()
    elif args.criterion == 'db_criterion':
        criterion = DB_Criterion()
    return criterion

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        if torch.cuda.is_available():
            # torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            # cudnn.deterministic = True
            # cudnn.benchmark = False
    else:
        # cudnn.benchmark = True
        pass

    trainTransform = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        ToTensor(),
        Normalize([0.], [1.])
        ])
    valTransform = Compose([
        ToTensor(),
        Normalize([0.], [1.])
    ])

    data_path = os.path.join(args.data_dir, args.dataset_name)
    data = split_data(data_path, nfolds=args.folds, expand_size=args.expand_size, random_state=args.seed)

    for fold in range(args.folds):
        # if fold not in [0]:
        #     break
        train_paths, val_paths = data[fold]

        trainset = labeledDataset(train_paths, trainTransform)
        valset = labeledDataset(val_paths, valTransform)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        model = getModel(args, device)
        if args.pretrained:
            snapshot = '/home/student/c2net/models/pretrained/loss_best_model.pth'
            rnn = GRU(input_dim=27, hidden_dim=32, layer_dim=2, output_dim=1).to(device)
            rnn.load_state_dict(torch.load(snapshot))
        else:
            rnn = None
        criterion = getCriterion(args)
        optimizer = optim.AdamW(model.parameters(), args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-5, patience=args.lr_patience)
        trainer = Trainer(args, model, criterion, optimizer, trainloader, valloader, device, scheduler=scheduler, fold=fold, rnn=rnn)
        trainer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)