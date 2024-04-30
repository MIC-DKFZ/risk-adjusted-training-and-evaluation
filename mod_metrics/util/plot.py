'''
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
SPDX-License-Identifier: BSD-3-Clause
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.ticker

def froc_plot(
        tpf: np.array, 
        fppi: np.array, 
        fpi_thresholds=[1/8,1/4,1/2,1,2,4,8], 
        title_start="FROC curve"
) -> matplotlib.figure.Figure:
    """
    Plot the FROC curve.
    @param tpf: True Positive Fraction
    @param fppi: False Positive Per Image
    @param fpi_thresholds: False Positive Per Image thresholds
    @param title_start: Title of the plot
    @returns: matplotlib.figure.Figure
    """
    ### Plot
    fig, ax = plt.subplots()
    ## y-axis
    ax.set_ylim([0, 1.02])
    plt.gca().yaxis.grid(True)
    plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    ## x-axis
    ax.set_xscale("log", base=2)
    curve = np.interp(fpi_thresholds, fppi, tpf)
    score = np.mean(curve)
    ax.plot(fpi_thresholds, curve, "o-")
    ax.set_title(f"{title_start} (score={score:.2f})")
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xticks(fpi_thresholds, rotation=33)
    return fig