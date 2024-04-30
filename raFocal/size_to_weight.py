'''
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
SPDX-License-Identifier: BSD-3-Clause
'''

from typing import Union, Dict, List, Tuple
from numpy import ndarray
from torch import Tensor
from numpy import float64 as np_float64
from torch import float64 as torch_float64
from numpy import where as np_where
from torch import where as torch_where
from numpy import exp as np_exp
from torch import exp as torch_exp
from numpy import poly1d as np_poly1d
from torch import from_numpy as torch_from_numpy
from nndet.core.boxes.ops import box_mad as torch_box_mad
from nndet.core.boxes.ops_np import box_mad as np_box_mad
from nndet.core.boxes.ops import box_mad_adjusted as torch_box_mad_adjusted
from nndet.core.boxes.ops_np import box_mad_adjusted as np_box_mad_adjusted
from nndet.core.boxes.ops import box_md_3D as torch_box_md_3D
from nndet.core.boxes.ops_np import box_md_3D as np_box_md_3D
from nndet.core.boxes.ops import box_md_3D_adjusted as torch_box_md_3D_adjusted
from nndet.core.boxes.ops_np import box_md_3D_adjusted as np_box_md_3D_adjusted
from nndet.core.boxes.ops import box_area as torch_box_area
from nndet.core.boxes.ops_np import box_area_np as np_box_area
from nndet.core.boxes.ops import box_area_adjusted as torch_box_area_adjusted
from nndet.core.boxes.ops_np import box_area_np_adjusted as np_box_area_adjusted

### ---- Size method names ----

SIZE_METHOD_MAX_AXIAL_DIAMETER = 'max-axial-diameter'
SIZE_METHOD_MAX_DIAMETER_3D = 'max-diameter-3d'
SIZE_METHOD_AREA = 'area'

### --- Size to weight function names ----

SIZE_TO_WEIGHT_FUNCTION_LINEAR = 'linear'
SIZE_TO_WEIGHT_FUNCTION_BANDPASS = 'bandpass'
SIZE_TO_WEIGHT_FUNCTION_BREAST_MORTALITY_RISK = 'breast-mortality-risk'
SIZE_TO_WEIGHT_FUNCTION_BREAST_MORTALITY_RISK_SOPIK = 'breast-mortality-risk-sopik'

### --- Size to weight functions ----

def _breast_mortality_risk(
    sizes: Union[Tensor, ndarray],
):
    """From: Modeling the Effect of Tumor Size in Early Breast Cancer,  Verschraegen et al., 2005"""
    is_torch = (isinstance(sizes, Tensor))
    exp = torch_exp if is_torch else np_exp
    return exp( -exp( -(sizes-15)/10 ) )

def _breast_mortality_risk_sopik(
    sizes: Union[Tensor, ndarray],
):
    """From: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6022519/"""
    is_torch = (isinstance(sizes, Tensor))
    p = np_poly1d([ 2.28e-07, -8.75e-05,  1.23e-02,  1.37e-03])
    if is_torch:
        res = p(sizes.detach().cpu().numpy())
    else:
        res = p(sizes)
    res = res / 0.641
    res = np_where(res>0.641, 0.641, res)
    if is_torch:
        res = torch_from_numpy(res)
        res = res.to(sizes.device, copy=False)
    return res


def _linear(
    sizes: Union[Tensor, ndarray], 
    min_weight: float = 0.5,
    max_weight: float = 3.0,
    low: int = 0,
    high: Union[int, str] = 'auto',
    apply_outside_range = True,
    decreasing = False,
) -> Union[Tensor, ndarray]:
    """Convert sizes to weights in a linear way

    Args:
        sizes (Union[Tensor, ndarray]): Input sizes [N]
        min_weight (float, optional): Weight to apply to minimum size. Defaults to 0.5.
        max_weight (float, optional): Weight to apply to maximum size. Defaults to 3.0.
        low (int, optional): Which value to consider as minimum. Defaults to 0.
        high (Union[int, str], optional): Which value to consider as maximum. 'auto' to use max of input. Defaults to 'auto'.
        apply_outside_range (bool, optional): Whether to apply min weight to values less than low/more than high, or make 0. Defaults to True.
        decreasing (bool, optional): Whether to reverse, so that high values go to min weight. Defaults to False.

    Returns:
        Union[Tensor, ndarray]: Resultant weights [N]
    """
    high = sizes.max() if high == 'auto' else high
    is_torch = (isinstance(sizes, Tensor))
    where = torch_where if is_torch else np_where
    weights = sizes.to(torch_float64) if is_torch else sizes.astype(np_float64)
    if not apply_outside_range:
        mask = where((weights<low)|(weights>high), False, True)
    ## Apply min-max
    weights[weights < low] = low
    weights[weights > high] = high
    ## To [0,1]
    weights -= float(low)
    weights /= float(high-low)
    if decreasing:
        weights *= -1.0
        weights += +1.0
    ## To [min_weight, max_weight]
    weights *= (max_weight-min_weight)
    weights += min_weight
    if not apply_outside_range:
        weights = where(mask, weights, 0.0)
    return weights

def _bandpass(
    sizes: Union[Tensor, ndarray], 
    min_weight: float = 0.5,
    max_weight: float = 3.0,
    low: int = 0,
    high: Union[int, str] = 'auto',
) -> Union[Tensor, ndarray]:
    """Convert sizes to weights like a band-pass filter.

    Args:
        sizes (Union[Tensor, ndarray]): Input sizes [N]
        min_weight (float, optional): Weight to apply to minimum size. Defaults to 0.5.
        max_weight (float, optional): Weight to apply to maximum size. Defaults to 3.0.
        low (int, optional): Which value to consider as minimum. Defaults to 0.
        high (Union[int, str], optional): Which value to consider as maximum. 'auto' to use max of input. Defaults to 'auto'.

    Returns:
        Union[Tensor, ndarray]: Resultant weights [N]
    """
    high = sizes.max() if high == 'auto' else high
    is_torch = (isinstance(sizes, Tensor))
    where = torch_where if is_torch else np_where
    weights = sizes.to(torch_float64) if is_torch else sizes.astype(np_float64)
    ## Apply min-max
    weights = where((weights < low) | (weights > high), min_weight, max_weight)
    return weights

_SIZE_TO_WEIGHT_NAME_TO_FUNCTION = {
    SIZE_TO_WEIGHT_FUNCTION_LINEAR : _linear,
    SIZE_TO_WEIGHT_FUNCTION_BANDPASS : _bandpass,
    SIZE_TO_WEIGHT_FUNCTION_BREAST_MORTALITY_RISK: _breast_mortality_risk,
    SIZE_TO_WEIGHT_FUNCTION_BREAST_MORTALITY_RISK_SOPIK: _breast_mortality_risk_sopik,
}

def boxes_size(
    boxes: Union[Tensor, ndarray], 
    size_method_name: str = SIZE_METHOD_AREA,
    x_spacings: Union[Tensor, ndarray, None] = None,
    y_spacings: Union[Tensor, ndarray, None] = None,
    z_spacings: Union[Tensor, ndarray, None] = None,
) -> Union[Tensor, ndarray]:
    """Calculate size of boxes given a method (possibilities specified above in this file)

    Args:
        boxes (Union[Tensor, ndarray]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2, z1, z2) format. [N, 6]
        size_method_name (str, optional): Method to be used for size calculation. Defaults to 'area'.
        x_spacings (Union[Tensor, ndarray, None], optional): physical spacing of the x-dimension [N]. 
            Without it physical size will not be considered.
        y_spacings (Union[Tensor, ndarray, None], optional): physical spacing of the y-dimension [N]
            Without it physical size will not be considered.
        z_spacings (Union[Tensor, ndarray, None], optional): physical spacing of the z-dimension [N]
            If 2D data then not provide.

    Raises:
        ValueError: Size method name is unknown
        NotImplementedError: Currently x_spacings,y_spacings is only used for max-axial-diameter.

    Returns:
        Union[Tensor, ndarray]: Resultant sizes [N]
    """
    is_torch = (isinstance(boxes, Tensor))
    if size_method_name == SIZE_METHOD_AREA:
        if x_spacings is not None and y_spacings is not None:
            sizes = torch_box_area_adjusted(boxes, x_spacings, y_spacings, z_spacings) if is_torch else np_box_area_adjusted(boxes, x_spacings, y_spacings, z_spacings)
        else:
            sizes = torch_box_area(boxes) if is_torch else np_box_area(boxes)
    elif size_method_name == SIZE_METHOD_MAX_AXIAL_DIAMETER:
        if x_spacings is not None and y_spacings is not None:
            sizes = torch_box_mad_adjusted(boxes, x_spacings, y_spacings) if is_torch else np_box_mad_adjusted(boxes, x_spacings, y_spacings)
        else:
            sizes = torch_box_mad(boxes) if is_torch else np_box_mad(boxes)
    elif size_method_name == SIZE_METHOD_MAX_DIAMETER_3D:
        if x_spacings is not None and y_spacings is not None and z_spacings is not None:
            sizes = torch_box_md_3D_adjusted(boxes, x_spacings, y_spacings, z_spacings) if is_torch else np_box_md_3D_adjusted(boxes, x_spacings, y_spacings, z_spacings)
        else:
            sizes = torch_box_md_3D(boxes) if is_torch else np_box_md_3D(boxes)
    else:
        raise ValueError(f'Unknown size_method_name: {size_method_name}')
    return sizes

def boxes_size_weight(
    boxes: Union[Tensor, ndarray], 
    size_method_name: str = SIZE_METHOD_AREA,
    size_to_weight_function_name: str = SIZE_TO_WEIGHT_FUNCTION_LINEAR,
    size_to_weight_function_config: Dict = {},
    x_spacings: Union[Tensor, ndarray, None] = None,
    y_spacings: Union[Tensor, ndarray, None] = None,
    z_spacings: Union[Tensor, ndarray, None] = None,
) -> Union[Tensor, ndarray]:
    """Calculate weights for bounding boxes, based on their size

    Args:
        boxes (Union[Tensor, ndarray]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2, z1, z2) format. [N, 6]
        size_method_name (str, optional): Method to be used for size calculation. Defaults to 'area'.
        size_to_weight_function_name (str, optional): Which function to use to calculate weight. Defaults to 'linear'.
        size_to_weight_function_config (Dict, optional): Configuration for the above function. Defaults to {}.
        x_spacings (Union[Tensor, ndarray, None], optional): physical spacing of the x-dimension [N]. 
            Without it physical size will not be considered.
        y_spacings (Union[Tensor, ndarray, None], optional): physical spacing of the y-dimension [N]
            Without it physical size will not be considered.
        z_spacings (Union[Tensor, ndarray, None], optional): physical spacing of the z-dimension [N]
            If 2D data then not provide.

    Returns:
        Union[Tensor, ndarray]: Resultant weights [N]
    """
    sizes = boxes_size(boxes, size_method_name, x_spacings, y_spacings, z_spacings)
    return _SIZE_TO_WEIGHT_NAME_TO_FUNCTION[size_to_weight_function_name](
        sizes, **size_to_weight_function_config
    )

def boxes_size_weight_combine(
    boxes: Union[Tensor, ndarray], 
    size_method_name: str = SIZE_METHOD_AREA,
    size_to_weight_function_name_config_tuples: List[Tuple[str, Dict]] = [(SIZE_TO_WEIGHT_FUNCTION_LINEAR, {})],
    x_spacings: Union[Tensor, ndarray, None] = None,
    y_spacings: Union[Tensor, ndarray, None] = None,
    z_spacings: Union[Tensor, ndarray, None] = None,
) -> Union[Tensor, ndarray]:
    """Calculate weights for bounding boxes, based on their size, 
    by combining different methodologies (addition).

    Args:
        boxes (Union[Tensor, ndarray]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2, z1, z2) format. [N, 6]
        size_method_name (str, optional): Method to be used for size calculation. Defaults to 'area'.
        size_to_weight_function_name_config_tuples (List[Tuple[str, Dict]], optional):
            List of size to weight function names+configs to apply.
            Defaults to [('linear', {})].
        x_spacings (Union[Tensor, ndarray, None], optional): physical spacing of the x-dimension [N]. 
            Without it physical size will not be considered.
        y_spacings (Union[Tensor, ndarray, None], optional): physical spacing of the y-dimension [N]
            Without it physical size will not be considered.
        z_spacings (Union[Tensor, ndarray, None], optional): physical spacing of the z-dimension [N]
            If 2D data then not provide.

    Returns:
        Union[Tensor, ndarray]: Resultant weights [N]
    """
    assert len(size_to_weight_function_name_config_tuples)
    ### Weight of first name+config
    weights = boxes_size_weight(
        boxes, size_method_name, 
        size_to_weight_function_name_config_tuples[0][0], size_to_weight_function_name_config_tuples[0][1],
        x_spacings, y_spacings, z_spacings,
    )
    ### Add the rest.
    for name, config in size_to_weight_function_name_config_tuples[1:]:
        weights += boxes_size_weight(
            boxes, size_method_name, 
            name, config,
            x_spacings, y_spacings, z_spacings,
        )
    return weights
