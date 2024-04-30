'''
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
SPDX-License-Identifier: BSD-3-Clause
'''

from typing import List, Dict
from dataclasses import dataclass, field

import numpy as np

from mod_metrics.representations.BoxData import BoxData, PredBoxData

@dataclass
class ModData:
    cases: List[str] = field(default_factory=list)
    case_to_gt_boxes: Dict[str, List[BoxData]] = field(default_factory=dict)
    case_to_pred_boxes: Dict[str, List[PredBoxData]] = field(default_factory=dict)
