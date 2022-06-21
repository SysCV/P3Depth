
import dataclasses
import math
from typing import Tuple
import torch
import numpy as np
from torch import Tensor

@dataclasses.dataclass
class DepthMetrics(object):
    silog: float = 0.
    rmse: float = 0.
    rmse_log: float = 0.
    sq_rel: float = 0.
    abs_rel: float = 0.
    lg10: float = 0.
    delta1: float = 0.
    delta2: float = 0.
    delta3: float = 0.

def compute_errors(gt, pred):

    pred = recover_metric_depth(pred, gt)

    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3


def recover_metric_depth(pred, gt):
    if type(pred).__module__ == torch.__name__:
        pred = pred.cpu().numpy()
    if type(gt).__module__ == torch.__name__:
        gt = gt.cpu().numpy()

    gt_mean = np.mean(gt)
    pred_mean = np.mean(pred)

    pred_metric = pred * (gt_mean / pred_mean)
    return pred_metric


def evaluate_depth_metrics(output, target, dataset_type, max_depth=10.0) -> DepthMetrics:

    _target = target.cpu().numpy()
    _output = output.cpu().numpy()

    min_depth_eval = 1e-3
    max_depth_eval = max_depth

    _output[_output < min_depth_eval] = min_depth_eval
    _output[_output > max_depth_eval] = max_depth_eval
    _output[np.isinf(_output)] = max_depth_eval
    _output[np.isnan(_output)] = min_depth_eval

    valid_mask = np.logical_and(_target > min_depth_eval, _target < max_depth_eval)

    ## Eigen eval
    if "KITTI" in dataset_type:
        _, _, gt_height, gt_width = _target.shape
        eval_mask = np.zeros(valid_mask.shape)
        eval_mask[ :, :,int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
        # eval_mask[ :, :, int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1 # GARG CROP
        valid_mask = np.logical_and(valid_mask, eval_mask)

    silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3=compute_errors(_target[valid_mask], _output[valid_mask])

    metrics = DepthMetrics(
        silog=float(silog),
        rmse=float(rmse),
        rmse_log=float(rmse_log),
        sq_rel=float(sq_rel),
        abs_rel=float(abs_rel),
        lg10=float(log10),
        delta1=float(d1),
        delta2=float(d2),
        delta3=float(d3)
    )

    return metrics
