'''
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
SPDX-License-Identifier: BSD-3-Clause
'''

from dataclasses import dataclass, field

import numpy as np
import matplotlib.figure

from mod_metrics.representations.BoxData import BoxData, PredBoxData
from mod_metrics.representations.FrocInputData import FrocInputData

@dataclass
class FrocOutputData:
    input_data: FrocInputData
    tpf: np.array # True positive fraction (sensitivity)
    fppi: np.array # False positive per image
    score: float # FROC score
    curve: np.array # FROC curve data
    fig: matplotlib.figure.Figure # matplotlib figure
