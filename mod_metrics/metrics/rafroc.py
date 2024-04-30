'''
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
SPDX-License-Identifier: BSD-3-Clause
'''

import argparse, copy
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.ticker

from mod_metrics.representations.FrocInputData import FrocInputData
from mod_metrics.representations.FrocOutputData import FrocOutputData

def rafroc(input_data: FrocInputData, weights: np.array, matching_gt_idxs: np.array, gt_weights: np.array) -> FrocOutputData:
    """
    Calculate FROC curve.
    @param input_data: FrocInputData
    @param weights: weight of each prediction (provided by some risk function)
    @param matching_gt_idxs: matching ground truth indexes for each prediction (if not matched, -1)
    @param gt_weights: weights of ground truth objects (provided by some risk function)
    @returns: FrocOutputData
    """
    Y_pred = input_data.Y_pred
    Y_true = input_data.Y_true
    S = input_data.S # number of cases
    fpi_thresholds = input_data.fpi_thresholds
    weights = copy.deepcopy(weights)

    ## Use GT weight if available
    for idx, match_idx in enumerate(matching_gt_idxs):
        if match_idx != -1:
            weights[idx] = gt_weights[match_idx]

    TPF  = np.zeros(len(set(Y_pred))+1, dtype=float) # -> always 0 at loc 0
    FPPI = np.zeros(len(set(Y_pred))+1, dtype=float) # -> always 0 at loc 0

    ## Normalize weights
    MAX_W = np.max(np.append(weights, gt_weights))
    weights /= MAX_W
    gt_weights /= MAX_W

    for idx, ksi in enumerate( sorted(set(Y_pred), reverse=True) ):
        _tp_probs = Y_pred[Y_true == 1]
        _fp_probs = Y_pred[Y_true == 0]
        _tp_weights = weights[Y_true == 1]
        _inv_fp_weights = 1 - weights[Y_true == 0]
        TPF[idx+1]  = sum(_tp_weights[_tp_probs >= ksi]) / float(sum(gt_weights))
        FPPI[idx+1] = sum(_inv_fp_weights[_fp_probs >= ksi]) / S
        
    ### Curve and score
    curve = np.interp(fpi_thresholds, FPPI, TPF)
    score = np.mean(curve)
    ### Plot
    fig, ax = plt.subplots()
    ax.set_ylim([0, 1.02])
    plt.gca().yaxis.grid(True)
    plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    ax.set_xscale("log", base=2)
    ax.plot(fpi_thresholds, curve, "o-")
    ax.set_title(f"raFROC curve (score={score:.2f})")
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xticks(fpi_thresholds, rotation=33)
    ### Result
    return FrocOutputData(
        input_data = input_data,
        tpf = TPF,
        fppi = FPPI,
        score = score,
        curve = curve,
        fig = fig
    )
