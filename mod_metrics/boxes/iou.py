'''
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
SPDX-License-Identifier: BSD-3-Clause
'''

from typing import List

import numpy as np

from mod_metrics.representations.BoxData import BoxData, PredBoxData

def iou_raw(gt_box: BoxData, pred_box: PredBoxData) -> float:
    """
    Calculate the IoU between two boxes.

    Args:
        gt_box: Ground truth box.
        pred_boxes: Predicted box.

    Returns:
        float: IoU value.
    """
    ### Intersection
    x1 = max(gt_box.x1, pred_box.x1)
    x2 = min(gt_box.x2, pred_box.x2)
    y1 = max(gt_box.y1, pred_box.y1)
    y2 = min(gt_box.y2, pred_box.y2)
    z1 = max(gt_box.z1, pred_box.z1)
    z2 = min(gt_box.z2, pred_box.z2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1) * max(0, z2 - z1)
    ### Union
    gt_volume = (gt_box.x2 - gt_box.x1) * (gt_box.y2 - gt_box.y1) * (gt_box.z2 - gt_box.z1)
    pred_volume = (pred_box.x2 - pred_box.x1) * (pred_box.y2 - pred_box.y1) * (pred_box.z2 - pred_box.z1)
    union = gt_volume + pred_volume - intersection
    ### IoU
    return intersection / union

def iou_raw_batch(gt_boxes: List[BoxData], pred_boxes: List[PredBoxData]) -> np.array:
    """
    Calculate the IoU between two lists of boxes.

    Args:
        gt_boxes: List of ground truth boxes.
        pred_boxes: List of predicted boxes.

    Returns:
        np.array: IoU values. Can be accessed by [gt_box_idx, pred_box_idx].
    """
    ### Initiate 2D array
    ious = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=float)
    ### Calculate IoU for each pair of boxes
    for gt_idx, gt_box in enumerate(gt_boxes):
        for pred_idx, pred_box in enumerate(pred_boxes):
            ious[gt_idx, pred_idx] = iou_raw(gt_box, pred_box)
    return ious

def __duplicate_nonzero_in_rows_or_cols(arr: np.array) -> bool:
    """Check whether there are rows or columns with multiple non-zero values in a 2D array

    Args:
        arr (np.array): input 2D array

    Returns:
        bool: Whether there are non-zero rows or cols
    """
    for row in arr:
        if len(row[row>0.0])>1:
            return True
    for column in arr.T:
        if len(column[column>0.0])>1:
            return True
    return False

def iou_raw_batch_preprocessed(
        gt_boxes: List[BoxData], pred_boxes: List[PredBoxData], threshold: float = 0.0
) -> np.array:
    """
    Calculate the IoU between two lists of boxes.
    Preprocess IoUs so that there is 1 value left per row/column, with the standard method.

    Args:
        gt_boxes: List of ground truth boxes.
        pred_boxes: List of predicted boxes.

    Returns:
        np.array: IoU values. Can be accessed by [gt_box_idx, pred_box_idx].
    """
    ### Calculate all IoUs
    ious = iou_raw_batch(gt_boxes, pred_boxes)
    ### Threshold
    ious[ious < threshold+1e-12] = 0.0
    ### Keep largest per row and per column
    xs = (np.array([x.prob for x in pred_boxes])).argsort() # sorted indexes of scores/probs
    while __duplicate_nonzero_in_rows_or_cols(ious):
        x, xs = xs[-1], xs[:-1] # pop highest score/prob
        y = ious[x].argmax() # GT index with highest IoU for this prediction
        value = ious[x][y]
        if value > 0+1e-12: # if there is indeed a successful prediction, remove others
            ious[x] = np.zeros(ious.shape[1])
            ious[:, y] = np.zeros(ious.shape[0])
            ious[x][y] = value
    return ious