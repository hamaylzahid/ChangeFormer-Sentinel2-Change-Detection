# put loss and metrics here
import torch
import torch.nn as nn

bce = nn.BCELoss()

def dice_loss(pred, target):
    smooth=1.
    pred,target = pred.view(-1), target.view(-1)
    inter = (pred*target).sum()
    return 1-(2*inter+smooth)/(pred.sum()+target.sum()+smooth)

def total_loss(pred, target):
    return bce(pred,target)+dice_loss(pred,target)

def compute_metrics(pred, mask):
    pred_bin = (pred>0.5).float()
    TP = (pred_bin*mask).sum()
    FP = (pred_bin*(1-mask)).sum()
    FN = ((1-pred_bin)*mask).sum()
    iou = TP/(TP+FP+FN+1e-6)
    precision = TP/(TP+FP+1e-6)
    recall = TP/(TP+FN+1e-6)
    f1 = 2*precision*recall/(precision+recall+1e-6)
    return iou.item(), precision.item(), recall.item(), f1.item()