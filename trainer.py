import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np
import logging
from datetime import datetime
from alive_progress import alive_it
from torchvision import transforms as transforms
from tensorboardX import SummaryWriter
from utils.plot import *


def attention(img_tensor, model=None, direction='W'):
    """
    args:
        img_tensor: a tensor with shape (B, 3, H, W);
        model: a pretrained RNN model;
    returns:
        guide_map: a tensor with shape (B, 1, H, W);
    """
    assert img_tensor.dim() == 4 and img_tensor.shape[1] == 3 # (B, 3, H, W)
    B, _, H, W = img_tensor.size()
    # print(img_tensor.shape)

    MIN, MAX = img_tensor.min(), img_tensor.max()
    if MAX != MIN:
        img_tensor = (img_tensor - MIN) / (MAX - MIN)

    pad_h, pad_w = 3 - H % 3, 3 - W % 3
    img_vol = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
    assert img_vol.shape[2] % 3 == 0 and img_vol.shape[3] % 3 == 0

    inputs_lst = []
    if direction == 'H':
        for b in range(B):
            for h in range(0, H, 3):
                img_chunk = img_vol[b, :, h: h + 3, :].transpose(0, 2)
                inputs_lst.append(img_chunk.reshape(-1, 27))
    elif direction == 'W':
        for b in range(B):
            for w in range(0, W, 3):
                img_chunk = img_vol[b, :, :, w: w + 3].transpose(0, 1)
                inputs_lst.append(img_chunk.reshape(-1, 27))

    inputs = torch.stack(inputs_lst)
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(inputs)).squeeze()

    step = preds.shape[0] // B
    lst = [preds[i: i + step] for i in range(0, preds.shape[0], step)]
    preds = torch.stack(lst)

    if direction == 'W':
        preds = preds.transpose(1, 2)

    preds = F.interpolate(preds.unsqueeze(1), size=[H, W], mode='nearest')
    # print(preds.shape)
    return preds


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, trainloader, valloader, device, scheduler=None, fold=None, rnn=None):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.rnn = rnn
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.scheduler = scheduler
        self.fold = fold
        self.accumulate_step = args.accumulate_step
        self.logging = Trainer.getLog(self.args)
        self.logging.info("====================\nArgs:{}\n====================".format(self.args))

    def getLog(args):
        dirname = os.path.join(args.logs_dir, args.arch, 'epochs_'+str(args.epochs),
        'batch_size_'+str(args.batch_size))
        filename = os.path.join(dirname, 'log.log')
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        logging.basicConfig(
                filename=filename,
                level=logging.INFO,
                format='%(asctime)s:%(message)s'
            )
        return logging
    
    def train(self):
        best_train_epoch, best_val_epoch = 0, 0
        best_train_dice, best_val_dice = 0., 0.
        train_loss_curve = list()
        valid_loss_curve = list()
        train_dice_curve = list()
        valid_dice_curve = list()

        dirname = os.path.join(self.args.logs_dir, self.args.arch, 'epochs_'+str(self.args.epochs),
        'batch_size_'+str(self.args.batch_size))
        now = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now())
        writer = SummaryWriter(os.path.join(dirname,now))
        iter_num = 0

        for epoch in alive_it(range(1, self.args.epochs + 1)):
            self.logging.info('-' * 20)
            self.logging.info('Epoch {}/{} lr: {}'.format(epoch, self.args.epochs, self.optimizer.param_groups[0]['lr']))
            self.logging.info('-' * 20)
            dt_size = len(self.trainloader.dataset)
            train_loss = 0
            train_dice = 0
            step = 0
            N = self.trainloader.batch_size
            small_batch_size = N // self.accumulate_step
            batch_num = (dt_size - 1) // N + 1

            # train
            self.model.train()
            for liver_imgs, tumor_mask in self.trainloader:
                step += 1
                liver_imgs = liver_imgs.to(self.device)
                tumor_mask = tumor_mask.to(self.device)
                batch = liver_imgs.size(0)

                self.optimizer.zero_grad()

                # rnn
                if self.rnn != None:
                    att_map_h = attention(liver_imgs, self.rnn, 'H')
                    att_map_w = attention(liver_imgs, self.rnn, 'W')
                    merge = torch.cat([att_map_h, att_map_w], dim=1)
                    att_map, _ = torch.max(merge, dim=1, keepdim=True)
                # print(att_map.shape)

                # forward -- accumulate gradient
                predicts, targets, metric_imgs, metric_targets= self.model(liver_imgs.float(), tumor_mask, att_map)
                loss = self.criterion(predicts, targets) / self.accumulate_step
                loss.backward()
                self.optimizer.step()
                
                # metric
                dice = self.dice_metric(metric_imgs, metric_targets)
                train_dice += dice.item()
                train_loss += loss.item()
                train_loss_curve.append(loss.item() / small_batch_size)

                self.logging.info("fold: %d, %d/%d, train_loss:%0.8f, train_dice:%0.8f" % (
                    self.fold, step, batch_num, loss.item(), dice.item() / batch))
                print("fold: %d, %d/%d, train_loss:%0.8f, train_dice:%0.8f" % (
                    self.fold, step, batch_num, loss.item(), dice.item() / batch))

                #tensorboard
                iter_num += 1
                writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], iter_num)
                writer.add_scalars('loss', {'loss': train_loss}, iter_num)
                writer.add_scalars('dice', {'dice': dice}, iter_num)

            aver_train_dice = train_dice / dt_size
            aver_train_loss = train_loss / batch_num
            train_dice_curve.append(aver_train_dice)

            self.logging.info("epoch %d aver_train_loss:%0.8f, aver_train_dice:%0.8f" % (epoch, aver_train_loss, aver_train_dice))
            print("epoch %d aver_train_loss:%0.8f, aver_train_dice:%0.8f" % (epoch, aver_train_loss, aver_train_dice))

            # Validate
            aver_val_loss, aver_val_dice = self.val_epoch()
            valid_loss_curve.append(aver_val_loss)
            
            # self.logging.info(type(aver_val_dice))
            valid_dice_curve.append(aver_val_dice)
            self.logging.info("epoch %d aver_valid_loss:%0.8f, aver_valid_dice:%0.8f" % (epoch, aver_val_loss, aver_val_dice))
            print("epoch %d aver_valid_loss:%0.8f, aver_valid_dice:%0.8f" % (epoch, aver_val_loss, aver_val_dice))

            # save model weight
            weights_path = os.path.join(self.args.weights_dir, self.args.arch,
                                        'batch_size_' + str(self.args.batch_size))
            if not os.path.exists(weights_path):
                os.makedirs(weights_path)

            if (epoch + 1) % 2 == 0:
                filename = 'fold_' + str(self.fold) + '_epochs_' + str(epoch + 1) + '.pth'
                weight_path = os.path.join(weights_path, filename)
                torch.save(self.model.state_dict(), weight_path)

            if best_train_dice < aver_train_dice:
                best_train_dice = aver_train_dice
                best_train_epoch = epoch

            if best_val_dice < aver_val_dice:
                best_val_weight_path = os.path.join(weights_path, 'fold_' + str(self.fold) + '_best_val_dice.pth')
                torch.save(self.model.state_dict(), best_val_weight_path)
                best_val_dice = aver_val_dice
                best_val_epoch = epoch

            self.logging.info("epoch:%d best_train_dice:%0.8f, best_train_epoch:%d, best_valid_dice:%0.8f, best_val_epoch:%d"
            % (epoch, best_train_dice, best_train_epoch, best_val_dice, best_val_epoch))

            # scheduler
            if self.scheduler is not None:
                self.scheduler.step(aver_val_dice)

        train_x = range(len(train_loss_curve))
        train_y = train_loss_curve

        train_iters = len(self.trainloader)
        valid_x = np.arange(1, len(valid_loss_curve) + 1) * train_iters
        valid_y = valid_loss_curve
        loss_plot(self.args, self.fold, train_x, train_y, valid_x, valid_y)
        metrics_plot(self.args, self.fold, 'train&valid', train_dice_curve, valid_dice_curve)

        writer.close()

    def val_epoch(self):
        save_root = self.args.predicts_dir
        self.model.eval()
        with torch.no_grad():
            loss_v, dice_v, ii = 0., 0., 0
            dt_size = len(self.valloader.dataset)
            batch_num = (dt_size - 1) // self.valloader.batch_size + 1

            # validation
            for liver_imgs, tumor_mask in self.valloader:
                liver_imgs = liver_imgs.to(self.device)
                tumor_mask = tumor_mask.to(self.device)

                if self.rnn != None:
                    att_map_h = attention(liver_imgs, self.rnn, 'H')
                    att_map_w = attention(liver_imgs, self.rnn, 'W')
                    merge = torch.cat([att_map_h, att_map_w], dim=1)
                    att_map, _ = torch.max(merge, dim=1, keepdim=True)

                predicts, targets, metric_imgs, metric_targets = self.model(liver_imgs, tumor_mask, att_map)
                loss = self.criterion(predicts, targets)

                loss_v += loss.item()
                dice_v += self.dice_metric(metric_imgs, metric_targets).cpu()
                metric_imgs = torch.sigmoid(metric_imgs)
                metric_imgs = metric_imgs * torch.round(metric_imgs)
                metric_targets = (metric_targets > 0).float()

                # save prediction
                for num in range(liver_imgs.shape[0]):
                    index = liver_imgs.shape[1] // 2
                    x = torch.squeeze(liver_imgs[num, index, :, :]).cpu().numpy()
                    output = torch.squeeze(metric_imgs[num, 0, :, :]).cpu().numpy()
                    gt = torch.squeeze(metric_targets[num, 0, :, :]).cpu().numpy()
                    # cv2.imshow('img', img_y)
                    src_path = os.path.join(save_root, "predict_%d_origin.png" % ii)
                    output_path = os.path.join(save_root, "predict_%d_predict.png" % ii)
                    gt_path = os.path.join(save_root, "predict_%d_mask.png" % ii)

                    cv2.imwrite(src_path, x * 255)
                    cv2.imwrite(output_path, output * 255)
                    cv2.imwrite(gt_path, gt * 255)
                    ii += 1

        return loss_v / batch_num, dice_v / dt_size

    def dice_metric(self, predicts, targets):
        smooth = 1e-5
        predicts = (predicts > 0).float().cpu()
        targets = (targets > 0).float().cpu()

        N = targets.size(0)
        pred_flat = predicts.view(N, -1)
        gt_flat = targets.view(N, -1)
        intersection = (pred_flat * gt_flat).sum(1)
        unionset = pred_flat.sum(1) + gt_flat.sum(1)
        dice = (2 * intersection + smooth) / (unionset + smooth)
        return dice.sum()