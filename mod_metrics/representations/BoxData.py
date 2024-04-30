'''
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
SPDX-License-Identifier: BSD-3-Clause
'''

from typing import List
from dataclasses import dataclass

@dataclass
class BoxData:
    x1: int
    x2: int
    y1: int
    y2: int
    z1: int
    z2: int

@dataclass
class PredBoxData(BoxData):
    prob: float

@dataclass
class BoxDataW(BoxData):
    weight: float

@dataclass
class PredBoxDataW(BoxData):
    prob: float
    weight: float