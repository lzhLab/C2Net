import os
import time
import inspect
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from PIL import Image
except ImportError:
    Image = None


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        val = float(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def binary_dice_score(logits, targets, threshold=0.5, eps=1e-7):
    """
    Compute binary Dice score from logits.

    logits:  [B, 1, H, W] or [B, H, W]
    targets: [B, 1, H, W] or [B, H, W]
    """
    if logits.dim() == 3:
        logits = logits.unsqueeze(1)

    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    targets = targets.float()

    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    dims = tuple(range(1, preds.dim()))
    intersection = torch.sum(preds * targets, dim=dims)
    union = torch.sum(preds, dim=dims) + torch.sum(targets, dim=dims)

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def binary_iou_score(logits, targets, threshold=0.5, eps=1e-7):
    """
    Compute binary IoU/Jaccard score from logits.
    """
    if logits.dim() == 3:
        logits = logits.unsqueeze(1)

    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    targets = targets.float()

    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    dims = tuple(range(1, preds.dim()))
    intersection = torch.sum(preds * targets, dim=dims)
    union = torch.sum(preds + targets, dim=dims) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def save_binary_prediction(logits, save_path):
    """
    Save the first prediction in a batch as a grayscale PNG.

    This function is optional and only used when PIL is available.
    """
    if Image is None:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with torch.no_grad():
        if logits.dim() == 4:
            logits = logits[0, 0]
        elif logits.dim() == 3:
            logits = logits[0]

        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).float()
        pred = pred.detach().cpu().numpy()
        pred = (pred * 255).astype(np.uint8)

        Image.fromarray(pred).save(save_path)


class Trainer:
    """
    Trainer for image segmentation.

    Compatible with:
    1. New C2Net/RNN_Model:
        output = model(image)

    2. Old model interface:
        predicts, targets, metric_imgs, metric_targets = model(image, mask, att_map)
    """

    def __init__(
        self,
        args,
        model,
        criterion,
        optimizer,
        trainloader,
        valloader,
        device,
        scheduler=None,
        fold=0,
        rnn=None
    ):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.scheduler = scheduler
        self.fold = fold
        self.rnn = rnn

        self.epochs = getattr(args, "epochs", 100)
        self.accumulate_step = max(1, int(getattr(args, "accumulate_step", 1)))

        self.weights_dir = getattr(args, "weights_dir", "results/weights")
        self.predicts_dir = getattr(args, "predicts_dir", "results/predicts")
        self.logs_dir = getattr(args, "logs_dir", "results/logs")

        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.predicts_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.best_val_dice = -1.0
        self.best_epoch = -1

    def make_attention_map(self, liver_imgs):
        """
        Build attention map for the old RNN-assisted model.

        For the new C2Net/RNN_Model, self.rnn should be None,
        so this function simply returns None.
        """
        if self.rnn is None:
            return None

        try:
            attention_func = globals().get("attention", None)

            if attention_func is None:
                raise RuntimeError(
                    "self.rnn is not None, but attention() function is not available "
                    "in trainer.py. Please import or define attention()."
                )

            att_map_h = attention_func(liver_imgs, self.rnn, "H")
            att_map_w = attention_func(liver_imgs, self.rnn, "W")

            merge = torch.cat([att_map_h, att_map_w], dim=1)
            att_map, _ = torch.max(merge, dim=1, keepdim=True)

            return att_map

        except Exception as exc:
            raise RuntimeError(
                "Failed to build attention map for the old RNN-assisted model."
            ) from exc

    def forward_model(self, liver_imgs, tumor_mask):
        """
        Compatible forward wrapper.

        New model:
            outputs = model(liver_imgs)

        Old model:
            predicts, targets, metric_imgs, metric_targets =
                model(liver_imgs, tumor_mask, att_map)
        """
        liver_imgs = liver_imgs.float()
        tumor_mask = tumor_mask.float()

        att_map = self.make_attention_map(liver_imgs)

        if att_map is not None:
            try:
                outputs = self.model(liver_imgs, tumor_mask, att_map)
            except TypeError:
                outputs = self.model(liver_imgs)
        else:
            try:
                outputs = self.model(liver_imgs)
            except TypeError:
                outputs = self.model(liver_imgs, tumor_mask, None)

        if isinstance(outputs, (tuple, list)):
            if len(outputs) == 4:
                predicts, targets, metric_imgs, metric_targets = outputs
            elif len(outputs) == 2:
                predicts, targets = outputs
                metric_imgs = predicts
                metric_targets = targets
            else:
                predicts = outputs[0]
                targets = tumor_mask
                metric_imgs = predicts
                metric_targets = tumor_mask
        else:
            predicts = outputs
            targets = tumor_mask
            metric_imgs = outputs
            metric_targets = tumor_mask

        targets = targets.float()
        metric_targets = metric_targets.float()

        if predicts.dim() == 3:
            predicts = predicts.unsqueeze(1)

        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        if metric_imgs.dim() == 3:
            metric_imgs = metric_imgs.unsqueeze(1)

        if metric_targets.dim() == 3:
            metric_targets = metric_targets.unsqueeze(1)

        return predicts, targets, metric_imgs, metric_targets

    def train(self):
        print("=" * 80)
        print(f"Start training fold {self.fold}")
        print("=" * 80)

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()

            train_loss, train_dice, train_iou = self.train_epoch(epoch)
            val_loss, val_dice, val_iou = self.val_epoch(epoch)

            if self.scheduler is not None:
                self._step_scheduler(val_loss, val_dice)

            is_best = val_dice > self.best_val_dice

            if is_best:
                self.best_val_dice = val_dice
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)

            self.save_checkpoint(epoch, is_best=False)

            elapsed = time.time() - start_time

            log_msg = (
                f"Fold [{self.fold}] "
                f"Epoch [{epoch:03d}/{self.epochs:03d}] "
                f"Time {elapsed:.1f}s | "
                f"Train Loss {train_loss:.4f} "
                f"Train Dice {train_dice:.4f} "
                f"Train IoU {train_iou:.4f} | "
                f"Val Loss {val_loss:.4f} "
                f"Val Dice {val_dice:.4f} "
                f"Val IoU {val_iou:.4f} | "
                f"Best Dice {self.best_val_dice:.4f} "
                f"at Epoch {self.best_epoch}"
            )

            print(log_msg)
            self.write_log(log_msg)

        print("=" * 80)
        print(
            f"Finished fold {self.fold}. "
            f"Best Val Dice: {self.best_val_dice:.4f} "
            f"at Epoch {self.best_epoch}"
        )
        print("=" * 80)

    def train_epoch(self, epoch):
        self.model.train()

        loss_meter = AverageMeter()
        dice_meter = AverageMeter()
        iou_meter = AverageMeter()

        self.optimizer.zero_grad(set_to_none=True)

        batch_num = len(self.trainloader)

        for step, batch in enumerate(self.trainloader, start=1):
            liver_imgs, tumor_mask = self.unpack_batch(batch)

            liver_imgs = liver_imgs.to(self.device, non_blocking=True)
            tumor_mask = tumor_mask.to(self.device, non_blocking=True)

            predicts, targets, metric_imgs, metric_targets = self.forward_model(
                liver_imgs,
                tumor_mask
            )

            loss = self.criterion(predicts, targets)
            loss_for_backward = loss / self.accumulate_step
            loss_for_backward.backward()

            if step % self.accumulate_step == 0 or step == batch_num:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                dice = binary_dice_score(metric_imgs, metric_targets)
                iou = binary_iou_score(metric_imgs, metric_targets)

            batch_size = liver_imgs.size(0)
            loss_meter.update(loss.item(), batch_size)
            dice_meter.update(dice.item(), batch_size)
            iou_meter.update(iou.item(), batch_size)

            if step % 20 == 0 or step == batch_num:
                print(
                    f"Train Epoch [{epoch:03d}] "
                    f"Step [{step:04d}/{batch_num:04d}] "
                    f"Loss {loss_meter.avg:.4f} "
                    f"Dice {dice_meter.avg:.4f} "
                    f"IoU {iou_meter.avg:.4f}"
                )

        return loss_meter.avg, dice_meter.avg, iou_meter.avg

    @torch.no_grad()
    def val_epoch(self, epoch):
        self.model.eval()

        loss_meter = AverageMeter()
        dice_meter = AverageMeter()
        iou_meter = AverageMeter()

        save_root = os.path.join(self.predicts_dir, f"fold_{self.fold}", f"epoch_{epoch}")
        os.makedirs(save_root, exist_ok=True)

        batch_num = len(self.valloader)

        for step, batch in enumerate(self.valloader, start=1):
            liver_imgs, tumor_mask = self.unpack_batch(batch)

            liver_imgs = liver_imgs.to(self.device, non_blocking=True)
            tumor_mask = tumor_mask.to(self.device, non_blocking=True)

            predicts, targets, metric_imgs, metric_targets = self.forward_model(
                liver_imgs,
                tumor_mask
            )

            loss = self.criterion(predicts, targets)
            dice = binary_dice_score(metric_imgs, metric_targets)
            iou = binary_iou_score(metric_imgs, metric_targets)

            batch_size = liver_imgs.size(0)
            loss_meter.update(loss.item(), batch_size)
            dice_meter.update(dice.item(), batch_size)
            iou_meter.update(iou.item(), batch_size)

            if step == 1:
                save_path = os.path.join(save_root, "prediction_sample.png")
                save_binary_prediction(metric_imgs, save_path)

            if step % 20 == 0 or step == batch_num:
                print(
                    f"Val Epoch [{epoch:03d}] "
                    f"Step [{step:04d}/{batch_num:04d}] "
                    f"Loss {loss_meter.avg:.4f} "
                    f"Dice {dice_meter.avg:.4f} "
                    f"IoU {iou_meter.avg:.4f}"
                )

        return loss_meter.avg, dice_meter.avg, iou_meter.avg

    def unpack_batch(self, batch):
        """
        Support common dataset returns:
            image, mask
            image, mask, extra_info...
        """
        if isinstance(batch, (tuple, list)):
            if len(batch) < 2:
                raise ValueError("Batch should contain at least image and mask.")
            liver_imgs = batch[0]
            tumor_mask = batch[1]
        else:
            raise ValueError(
                "Unsupported batch format. Expected tuple/list: (image, mask, ...)."
            )

        return liver_imgs, tumor_mask

    def _step_scheduler(self, val_loss, val_dice):
        """
        Step scheduler safely.

        Recommended:
        - If ReduceLROnPlateau(mode='max'), monitor val_dice.
        - If ReduceLROnPlateau(mode='min'), monitor val_loss.
        """
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            mode = getattr(self.scheduler, "mode", "min")

            if mode == "max":
                self.scheduler.step(val_dice)
            else:
                self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def save_checkpoint(self, epoch, is_best=False):
        model_state = self.model.module.state_dict() if isinstance(
            self.model,
            nn.DataParallel
        ) else self.model.state_dict()

        checkpoint = {
            "epoch": epoch,
            "fold": self.fold,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_dice": self.best_val_dice,
            "best_epoch": self.best_epoch
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        latest_path = os.path.join(
            self.weights_dir,
            f"fold_{self.fold}_latest.pth"
        )

        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = os.path.join(
                self.weights_dir,
                f"fold_{self.fold}_best.pth"
            )
            torch.save(checkpoint, best_path)

    def write_log(self, message):
        log_path = os.path.join(
            self.logs_dir,
            f"fold_{self.fold}_train.log"
        )

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
