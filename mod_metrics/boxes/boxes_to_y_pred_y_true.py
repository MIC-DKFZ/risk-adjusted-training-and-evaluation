'''
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
SPDX-License-Identifier: BSD-3-Clause
'''

from typing import List, Tuple

import numpy as np

from mod_metrics.boxes.iou import iou_raw_batch_preprocessed
from mod_metrics.representations.BoxData import BoxData, PredBoxData
from mod_metrics.representations.ModData import ModData

def boxes_to_y_pred_y_true(
        gt_boxes: List[BoxData], pred_boxes: List[PredBoxData], iou_threshold = 0.1
) -> Tuple[np.array, np.array, np.array]:
    """
    Convert ground truth boxes and prediction boxes to y_pred and y_true for FROC calculation.

    Args:
        gt_boxes: List of ground truth BoxData objects.
        pred_boxes: List of PredBoxData objects.
        iou_threshold: IoU threshold for a prediction to be considered a true positive.

    Returns:
        Tuple[np.array, np.array, np.array]: Y_pred, Y_true, matching_gt_idx
    """
    ious = iou_raw_batch_preprocessed(gt_boxes, pred_boxes)

    Y_pred = np.array([pred_box.prob for pred_box in pred_boxes])
    
    Y_true = np.zeros(len(pred_boxes), dtype=int)

    matching_gt_idxs = np.zeros(len(pred_boxes), dtype=int)
    matching_gt_idxs[matching_gt_idxs == 0] = -1  

    for pred_idx, pred_box in enumerate(pred_boxes):
        for gt_idx, gt_box in enumerate(gt_boxes):
            if ious[gt_idx, pred_idx] >= iou_threshold:
                Y_true[pred_idx] = 1
                matching_gt_idxs[pred_idx] = gt_idx
                break

    return Y_pred, Y_true, matching_gt_idxs

def boxes_to_y_pred_y_true_batch(
        mod_data: ModData, iou_threshold = 0.1
) -> Tuple[np.array, np.array, np.array]:
    """
    Convert ground truth boxes and prediction boxes to Y_pred and Y_true for FROC calculation.

    Args:
        mod_data: ModData object.
        iou_threshold: IoU threshold for a prediction to be considered a true positive.

    Returns:
        Tuple[np.array, np.array]: Y_pred, Y_true
    """
    Y_pred = []
    Y_true = []
    matching_gt_idxs = []
    counter_gts_passed = 0
    for case in mod_data.cases:
        gt_boxes = mod_data.case_to_gt_boxes[case]
        pred_boxes = mod_data.case_to_pred_boxes[case]
        _y_pred, _y_true, _matching_gt_idxs = boxes_to_y_pred_y_true(gt_boxes, pred_boxes, iou_threshold)
        Y_pred.extend(_y_pred)
        Y_true.extend(_y_true)
        for _idx in _matching_gt_idxs:
            if _idx != -1:
                matching_gt_idxs.append( _idx + counter_gts_passed )
            else:
                matching_gt_idxs.append(-1)
        counter_gts_passed += len(gt_boxes)

    return np.array(Y_pred), np.array(Y_true), np.array(matching_gt_idxs)