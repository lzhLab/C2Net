import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
except ImportError:
    binary_erosion = None
    distance_transform_edt = None


try:
    from skimage.morphology import skeletonize
except ImportError:
    skeletonize = None


class DB_Criterion(nn.Module):
    """
    Loss function compatible with both single-output and deep-supervision outputs.

    Supported model outputs:
        1. Single output:
            predicts = pred

        2. Deep supervision output:
            predicts = (pred_1, pred_2)
            predicts = [pred_1, pred_2]

    For single-output C2Net/RNN_Model, this becomes standard BCEWithLogitsLoss.
    """

    def __init__(self):
        super(DB_Criterion, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predicts, targets):
        targets = targets.float()

        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        # New C2Net/RNN_Model case: predicts is a single tensor.
        if torch.is_tensor(predicts):
            if predicts.dim() == 3:
                predicts = predicts.unsqueeze(1)
            return self.criterion(predicts, targets)

        # Old deep-supervision case: predicts is tuple/list.
        if isinstance(predicts, (tuple, list)):
            if len(predicts) == 1:
                pred_1 = predicts[0]

                if pred_1.dim() == 3:
                    pred_1 = pred_1.unsqueeze(1)

                return self.criterion(pred_1, targets)

            pred_1, pred_2 = predicts[0], predicts[1]

            if pred_1.dim() == 3:
                pred_1 = pred_1.unsqueeze(1)

            if pred_2.dim() == 3:
                pred_2 = pred_2.unsqueeze(1)

            target_2 = F.adaptive_max_pool2d(targets, pred_2.shape[2:])

            loss1 = self.criterion(pred_1, targets)
            loss2 = self.criterion(pred_2, target_2)

            return 0.5 * (loss1 + loss2)

        raise TypeError(
            "Unsupported predicts type. Expected Tensor, tuple, or list, "
            f"but got {type(predicts)}."
        )


def _to_numpy_binary(pred, target, threshold=0.5):
    """
    Convert logits/probabilities and targets to binary numpy arrays.

    pred can be logits. Sigmoid is applied before thresholding.
    """
    if torch.is_tensor(pred):
        pred = pred.detach().float().cpu()

        if pred.dim() == 4:
            pred = pred[:, 0]
        elif pred.dim() == 3:
            pred = pred
        elif pred.dim() == 2:
            pred = pred.unsqueeze(0)

        pred = torch.sigmoid(pred)
        pred = (pred > threshold).numpy().astype(np.bool_)

    if torch.is_tensor(target):
        target = target.detach().float().cpu()

        if target.dim() == 4:
            target = target[:, 0]
        elif target.dim() == 3:
            target = target
        elif target.dim() == 2:
            target = target.unsqueeze(0)

        target = (target > 0.5).numpy().astype(np.bool_)

    return pred, target


def dice_score_np(pred, target, eps=1e-7):
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)

    intersection = np.logical_and(pred, target).sum()
    denominator = pred.sum() + target.sum()

    if denominator == 0:
        return 1.0

    return float((2.0 * intersection + eps) / (denominator + eps))


def jaccard_index_np(pred, target, eps=1e-7):
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    if union == 0:
        return 1.0

    return float((intersection + eps) / (union + eps))


def sensitivity_np(pred, target, eps=1e-7):
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)

    tp = np.logical_and(pred, target).sum()
    fn = np.logical_and(~pred, target).sum()

    if target.sum() == 0:
        return 1.0 if pred.sum() == 0 else 0.0

    return float((tp + eps) / (tp + fn + eps))


def _safe_skeletonize(mask):
    mask = mask.astype(np.bool_)

    if skeletonize is None:
        return mask

    return skeletonize(mask).astype(np.bool_)


def cldice_np(pred, target, eps=1e-7):
    """
    Compute clDice for binary 2D masks.

    clDice = 2 * Tprec * Tsens / (Tprec + Tsens)
    """
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)

    if pred.sum() == 0 and target.sum() == 0:
        return 1.0

    if pred.sum() == 0 or target.sum() == 0:
        return 0.0

    skel_pred = _safe_skeletonize(pred)
    skel_target = _safe_skeletonize(target)

    tprec_den = skel_pred.sum()
    tsens_den = skel_target.sum()

    if tprec_den == 0 or tsens_den == 0:
        return dice_score_np(pred, target, eps=eps)

    tprec = np.logical_and(skel_pred, target).sum() / (tprec_den + eps)
    tsens = np.logical_and(skel_target, pred).sum() / (tsens_den + eps)

    return float((2.0 * tprec * tsens + eps) / (tprec + tsens + eps))


def _surface(mask):
    mask = mask.astype(np.bool_)

    if binary_erosion is None:
        raise ImportError(
            "scipy is required for ASSD and HD. Please install scipy."
        )

    if mask.sum() == 0:
        return mask

    eroded = binary_erosion(mask)
    return np.logical_xor(mask, eroded)


def _surface_distances(pred, target):
    if distance_transform_edt is None:
        raise ImportError(
            "scipy is required for ASSD and HD. Please install scipy."
        )

    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)

    if pred.sum() == 0 and target.sum() == 0:
        return np.array([0.0]), np.array([0.0])

    if pred.sum() == 0 or target.sum() == 0:
        return np.array([np.nan]), np.array([np.nan])

    pred_surface = _surface(pred)
    target_surface = _surface(target)

    if pred_surface.sum() == 0 or target_surface.sum() == 0:
        return np.array([np.nan]), np.array([np.nan])

    dist_to_target = distance_transform_edt(~target_surface)
    dist_to_pred = distance_transform_edt(~pred_surface)

    pred_to_target = dist_to_target[pred_surface]
    target_to_pred = dist_to_pred[target_surface]

    return pred_to_target, target_to_pred


def assd_np(pred, target):
    pred_to_target, target_to_pred = _surface_distances(pred, target)

    distances = np.concatenate([pred_to_target, target_to_pred])

    if np.isnan(distances).all():
        return np.nan

    return float(np.nanmean(distances))


def hd_np(pred, target):
    pred_to_target, target_to_pred = _surface_distances(pred, target)

    distances = np.concatenate([pred_to_target, target_to_pred])

    if np.isnan(distances).all():
        return np.nan

    return float(np.nanmax(distances))


def compute_segmentation_metrics(preds, targets, threshold=0.5):
    """
    Compute per-sample segmentation metrics.

    Args:
        preds: logits, shape [B, 1, H, W] or [B, H, W]
        targets: binary masks, shape [B, 1, H, W] or [B, H, W]

    Returns:
        dict[str, list[float]]
    """
    preds_np, targets_np = _to_numpy_binary(
        preds,
        targets,
        threshold=threshold
    )

    results = {
        "DSC": [],
        "clDice": [],
        "JI": [],
        "Sensitivity": [],
        "ASSD": [],
        "HD": []
    }

    for pred, target in zip(preds_np, targets_np):
        results["DSC"].append(dice_score_np(pred, target))
        results["clDice"].append(cldice_np(pred, target))
        results["JI"].append(jaccard_index_np(pred, target))
        results["Sensitivity"].append(sensitivity_np(pred, target))
        results["ASSD"].append(assd_np(pred, target))
        results["HD"].append(hd_np(pred, target))

    return results


def mean_ci95(values):
    """
    Return mean and 95% confidence interval half-width.

    NaN values are ignored.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan, np.nan

    mean = float(np.mean(values))

    if len(values) == 1:
        return mean, 0.0

    sd = float(np.std(values, ddof=1))
    ci95 = 1.96 * sd / math.sqrt(len(values))

    return mean, ci95


def format_mean_ci95(values, precision=4):
    mean, ci95 = mean_ci95(values)

    if np.isnan(mean):
        return "nan"

    return f"{mean:.{precision}f}±{ci95:.{precision}f}"


class SegMetricMeter:
    """
    Accumulate segmentation metrics and report mean±95%CI.

    Example:
        meter = SegMetricMeter()
        meter.update(logits, masks)
        print(meter.summary_text())
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.records = {
            "DSC": [],
            "clDice": [],
            "JI": [],
            "Sensitivity": [],
            "ASSD": [],
            "HD": []
        }

    def update(self, preds, targets):
        batch_metrics = compute_segmentation_metrics(
            preds,
            targets,
            threshold=self.threshold
        )

        for key, values in batch_metrics.items():
            self.records[key].extend(values)

    def summary_dict(self, precision=4):
        return {
            key: format_mean_ci95(values, precision=precision)
            for key, values in self.records.items()
        }

    def summary_text(self, precision=4):
        summary = self.summary_dict(precision=precision)

        return (
            f"DSC↑ {summary['DSC']} | "
            f"clDice↑ {summary['clDice']} | "
            f"JI↑ {summary['JI']} | "
            f"Sensitivity↑ {summary['Sensitivity']} | "
            f"ASSD↓ {summary['ASSD']} | "
            f"HD↓ {summary['HD']}"
        )

    def raw(self):
        return self.records

