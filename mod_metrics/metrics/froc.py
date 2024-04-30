'''
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
SPDX-License-Identifier: BSD-3-Clause
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker

from mod_metrics.representations.FrocInputData import FrocInputData
from mod_metrics.representations.FrocOutputData import FrocOutputData

def froc(input_data: FrocInputData) -> FrocOutputData:
    """
    Calculate FROC curve.
    @param input_data: FrocInputData
    @returns: FrocOutputData
    """
    Y_pred = input_data.Y_pred
    Y_true = input_data.Y_true
    S = input_data.S
    T = input_data.T
    fpi_thresholds = input_data.fpi_thresholds
    ### Populate TPF (True Positive Fraction) and FPPI (False Positive Per Image)
    TPF  = np.zeros(len(set(Y_pred))+1, dtype=float) # -> always 0 at loc 0
    FPPI = np.zeros(len(set(Y_pred))+1, dtype=float) # -> always 0 at loc 0
    for idx, ksi in enumerate( sorted(set(Y_pred), reverse=True) ):
        _tp_probs = Y_pred[Y_true == 1]
        _fp_probs = Y_pred[Y_true == 0]
        TPF[idx+1]  = len(_tp_probs[_tp_probs >= ksi]) / T
        FPPI[idx+1] = len(_fp_probs[_fp_probs >= ksi]) / S
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
    ax.set_title(f"FROC curve (score={score:.2f})")
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
