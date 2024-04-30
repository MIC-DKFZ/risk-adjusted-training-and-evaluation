'''
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
SPDX-License-Identifier: BSD-3-Clause
'''

from dataclasses import dataclass, field

import numpy as np

from mod_metrics.representations.BoxData import BoxData, PredBoxData

@dataclass
class FrocInputData:
    S: int # Number of cases
    T: int # Number of ground truth objects
    Y_pred: np.array # Predictions
    Y_true: np.array # Ground truth
    fpi_thresholds: np.array = np.array([])

    def __post_init__(self):
        if self.fpi_thresholds.size == 0:
            self.fpi_thresholds = np.array([0.125, 0.25, 0.5, 1, 2, 4, 8])
