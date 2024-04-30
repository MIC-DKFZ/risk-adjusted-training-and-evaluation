import argparse

import numpy as np
import pandas as pd

def froc_independent(
    fp_probs: np.ndarray,
    tp_probs: np.ndarray,
    num_targets: int,
    num_images: int,
):
    """
    This function is modified from the official evaluation code of
    `CAMELYON 16 Challenge <https://camelyon16.grand-challenge.org/>`_, and used to compute
    the required data for plotting the Free Response Operating Characteristic (FROC) curve.

    Args:
        fp_probs: an array that contains the probabilities of the false positive detections for all
            images under evaluation.
        tp_probs: an array that contains the probabilities of the True positive detections for all
            images under evaluation.
        num_targets: the total number of targets (excluding `labels_to_exclude`) for all images under evaluation.
        num_images: the number of images under evaluation.

    """
    total_fps, total_tps = [0], [0]
    all_probs = sorted(set(list(fp_probs) + list(tp_probs)), reverse=True)
    for thresh in all_probs:
        total_fps.append((fp_probs >= thresh).sum())
        total_tps.append((tp_probs >= thresh).sum())
    fps_per_image = np.asarray(total_fps) / float(num_images)
    total_sensitivity = np.asarray(total_tps) / float(num_targets)
    return fps_per_image, total_sensitivity


if __name__=="__main__":
    ### Args
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str, help="Path to csv")
    parser.add_argument("S", type=int, help="Number of subjects")
    parser.add_argument("T", type=int, help="Number of ground truth objects")
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    S = args.S
    T = args.T
    fpi_thresholds = [1/8,1/4,1/2,1,2,4,8]
    ### Run
    Y_pred = np.array( df["Y_pred"].tolist() )
    Y_true = np.array( df["Y_true"].tolist() )
    print(">> Calculating...")
    fp_probs = Y_pred[Y_true == 0]
    tp_probs = Y_pred[Y_true == 1]
    fppi, tpf = froc_independent(fp_probs, tp_probs, T, S)
    print(">> Finished.")
    _idxs = np.argsort(fppi)
    fppi = fppi[_idxs]
    tpf = tpf[_idxs]
    print(fppi)
    print(tpf)
    ### Show score
    curve = np.interp(fpi_thresholds, fppi, tpf)
    score = np.mean(curve)
    print(f">> Score {score}")