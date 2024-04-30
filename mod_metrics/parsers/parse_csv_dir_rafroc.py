'''
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
SPDX-License-Identifier: BSD-3-Clause
'''

from pathlib import Path

import numpy as np
import pandas as pd

from mod_metrics.representations.ModData import ModData
from mod_metrics.representations.BoxData import BoxDataW, PredBoxDataW

def parse_csv_dir_rafroc(csv_dir: Path) -> ModData:
    """
    Parse a directory of csv files containing the results of a FROC analysis.

    Args:
        csv_dir: Path to the directory containing the csv files.

    Returns:
        A list of tuples, where each tuple contains the name of the csv file and the FROC score.
    """

    ### Read cases
    cases = list(set( pd.read_csv(csv_dir / "cases.csv")['Name'].tolist() ))
    ### Read gt
    case_to_gt_boxes = {x: [] for x in cases}
    for idx, row in pd.read_csv(csv_dir / "gt.csv").iterrows():
        case_to_gt_boxes[ row["Name"] ].append(
            BoxDataW(
                x1 = row["x1"],
                x2 = row["x2"],
                y1 = row["y1"],
                y2 = row["y2"],
                z1 = row["z1"],
                z2 = row["z2"],
                weight = row["Risk"],
            )
        )
    ### Read predictions
    case_to_pred_boxes = {x: [] for x in cases}
    for idx, row in pd.read_csv(csv_dir / "pred.csv").iterrows():
        case_to_pred_boxes[ row["Name"] ].append(
            PredBoxDataW(
                x1 = row["x1"],
                x2 = row["x2"],
                y1 = row["y1"],
                y2 = row["y2"],
                z1 = row["z1"],
                z2 = row["z2"],
                prob = row["prob"],
                weight = row["Risk"],
            )
        )    

    return ModData(
        cases = cases,
        case_to_gt_boxes = case_to_gt_boxes,
        case_to_pred_boxes = case_to_pred_boxes
    )