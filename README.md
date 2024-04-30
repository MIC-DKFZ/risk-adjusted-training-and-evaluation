<!-- 
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
SPDX-License-Identifier: BSD-3-Clause
 -->

# Risk-adjusted Medical Object Detection

Copyright German Cancer Research Center (DKFZ) and contributors. Please make sure that your usage of this code is in compliance with its [license](./LICENSE).

### Overview

Easy to use and reusable implementations of 

* FROC (Free-Response Operating Characteristic) metric
* raFROC (risk-adjusted FROC) metric
* raFocal (risk-adjusted focal loss)


### Setup

Requires python 3.9+. 

```bash
pip install -r requirements.txt
pip install -e .
```

Now the following should work:

```
# The cli
mod_metrics_v1 -h 
# The package
python -c "import mod_metrics"
```

### Reuse

* [mod_metrics/](./mod_metrics/) directory is a python package in order to be easily reused.
* raFocal works with nnDetection, see instructions below.

### Metrics

##### Using the terminal

Metrics can also be calculated without writing any code. Input data need to be arranged in spreadsheets (.csv) in a directory following the format specified.

###### FROC example

Input data directory should follow the format as in `example_csv_dir_froc/`

* `cases.csv` 
    * Column "Name": All the case IDs.
* `gt.csv` 
    * Column "Name": Case ID
    * Columns "x1","x2","y1","y2","z1","z2": bounding box coordinates. It doesn't matter which dimension is considered which, as long as it is consistent across all data provided.
* `pred.csv` 
    * Column "Name": Case ID
    * Columns "x1","x2","y1","y2","z1","z2": bounding box coordinates. It doesn't matter which dimension is considered which, as long as it is consistent across all data provided.
    * Column "prob": The model's predicted probability for the bounding box.

```
mod_metrics_v1 froc example_csv_dir_froc/ out/froc/
```

* NOTE: Data in the example dir are just to showcase the format and are too few to properly understand how changes affect the result. Please use your own dataset for that.

###### raFROC example

Input data directory should follow the format as in `example_csv_dir_rafroc/`. The format is the same as in FROC, with the differences:

* In `gt.csv` there is an extra column "Risk" which holds the calculated weight/risk for the ground truth object based on the risk function.
* In `pred.csv` there is an extra column "Risk" which holds the calculated weight/risk for the prediction based on the risk function. This should be based on characteristics of the prediction without necessarily knowing any underlying ground truth.

```
mod_metrics_v1 rafroc example_csv_dir_rafroc/ out/rafroc/
```

* NOTE: Data in the example dir are just to showcase the format and are too few to properly understand how changes affect the result. Please use your own dataset for that.

### Risk-adjusted focal loss (raFocal)

The example in [raFocal/anchor_RA.py](./raFocal/anchor_RA.py) is designed to be used with [nnDetection](https://github.com/MIC-DKFZ/nnDetection) (Reproducible with commit `b2e0e4829f4dd9662a2ea7df219a658589716f57` but should also work with latest).

* Place the file `raFocal/anchor_RA.py` (just the file not the raFocal dir) under nnDetection's `nndet/arch/heads/comb`
* Place the file `raFocal/size_to_weight.py` (just the file not the raFocal dir) under nnDetection's `nndet/arch/heads/comb`
* Place the config file `raFocal/ra_config.yaml` (just the file not the raFocal dir) under nnDetection's `nndet/conf/train`
* Replace x_training_spacing, y_training_spacing, z_training_spacing with the respective image spacing nnDetection uses in training (should be visible when you preprocess the data and in the fingerprint).
* Add to `nndet/arch/heads/comb/__init__.py`: 
    ```python
    from nndet.arch.heads.comb.anchor_RA import BoxHeadRAFocal
    ```
* Add to `nndet/ptmodule/retinaunet/dev/c011.py`: 
    ```python
    @MODULE_REGISTRY.register
    class RetinaUNetC011FocalRA(RetinaUNetC011Focal):
        """Same as RetinaUNetC011Focal but with risk-adjusted focal loss."""
        head_cls = BoxHeadRAFocal
    ```
* Lastly, in your `nndet_train` command, add `train=ra_config`
* Now you are using raFocal!
