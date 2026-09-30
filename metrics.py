import numpy as np
import torch
import torch.nn.functional as F

from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision


def _hausdorff_metrics(output, target):
    output_nonempty = np.any(output)
    target_nonempty = np.any(target)

    if output_nonempty and target_nonempty:
        return hd(output, target), hd95(output, target)
    if not output_nonempty and not target_nonempty:
        return 0.0, 0.0

    spatial_shape = np.asarray(output.shape[-2:], dtype=np.float64)
    max_distance = float(np.linalg.norm(np.maximum(spatial_shape - 1.0, 0.0)))
    return max_distance, max_distance



def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)

    _, hd95_ = _hausdorff_metrics(output_, target_)
    
    return iou, dice, hd95_


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def indicators(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)
    hd_, hd95_ = _hausdorff_metrics(output_, target_)
    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)

    return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_
