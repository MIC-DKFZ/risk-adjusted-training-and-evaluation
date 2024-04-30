'''
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
SPDX-License-Identifier: BSD-3-Clause
'''

import sys, argparse, pickle, json
from pathlib import Path

import numpy as np

from mod_metrics.parsers.parse_csv_dir_froc import parse_csv_dir_froc
from mod_metrics.parsers.parse_csv_dir_rafroc import parse_csv_dir_rafroc
from mod_metrics.representations.FrocInputData import FrocInputData
from mod_metrics.metrics.froc import froc
from mod_metrics.metrics.rafroc import rafroc
from mod_metrics.boxes.boxes_to_y_pred_y_true import boxes_to_y_pred_y_true_batch


############
### MAIN ###
############

def main():
    sys.path.append(Path(__file__).resolve().parent.parent.as_posix())

    parser = argparse.ArgumentParser(
                description="Object detection metrics. Check each subcommand with '-h' for more help.", 
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")

    ### ---- FROC Args ---- ###
    froc_parser = subparsers.add_parser("froc", help="FROC metric", description="FROC metric")
    froc_parser.add_argument("input_csv_dir", type=str, help="Path to the input csv dir")
    froc_parser.add_argument("output_dir", type=str, help="Path to the output dir")

    ### ---- raFROC Args ---- ###
    rafroc_parser = subparsers.add_parser("rafroc", help="raFROC metric", description="risk-adjusted FROC (raFROC) metric")
    rafroc_parser.add_argument("input_csv_dir", type=str, help="Path to the input csv dir")
    rafroc_parser.add_argument("output_dir", type=str, help="Path to the output dir")

    ### ---- Parse Args ---- ###
    args = parser.parse_args()

    ### ---- FROC command ---- ###
    if args.command == "froc":
        input_csv_dir = Path(args.input_csv_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Input data
        mod_data = parse_csv_dir_froc(input_csv_dir)
        Y_pred, Y_true, matching_gt_idxs = boxes_to_y_pred_y_true_batch(mod_data)
        froc_input_data = FrocInputData(
            S = int(len(mod_data.cases)), # number of cases
            T = sum( [len(x) for x in mod_data.case_to_gt_boxes.values()] ), # number of ground truth objects
            Y_pred = Y_pred,
            Y_true = Y_true,
        )
        # Run
        froc_output_data = froc(froc_input_data)
        # Save
        with open(output_dir / "froc_output_data.pkl", "wb") as f:
            pickle.dump(froc_output_data, f)
        with open(output_dir / "froc_score.json", "w") as f:
            json.dump({
                "score": froc_output_data.score,
            }, f, indent=4)
        froc_output_data.fig.savefig(output_dir / "froc.png")
        print(sorted(set(mod_data.cases)))
        print(f">> WARNING: You are using example data. Expect weirdness in behavior (e.g. large changes in result) compared to real data. ") if sorted(set(mod_data.cases)) == ["Patient1", "Patient2", "Patient3", "Patient4", "Patient5", "Patient6", "Patient7", "Patient8"] else None
        print(f">> FROC score: {froc_output_data.score}")
        print(f">> Done. Results saved at {str(output_dir.resolve())}")
    
    ### ---- raFROC command ---- ###
    elif args.command == "rafroc":
        input_csv_dir = Path(args.input_csv_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Input data
        mod_data = parse_csv_dir_rafroc(input_csv_dir)
        Y_pred, Y_true, matching_gt_idxs = boxes_to_y_pred_y_true_batch(mod_data)
        weights = []
        gt_weights = []
        for case in mod_data.cases:
            weights.extend([box.weight for box in mod_data.case_to_pred_boxes[case]])
            gt_weights.extend([box.weight for box in mod_data.case_to_gt_boxes[case]])
        weights = np.array(weights)
        gt_weights = np.array(gt_weights)
        froc_input_data = FrocInputData(
            S = int(len(mod_data.cases)), # number of cases
            T = sum( [len(x) for x in mod_data.case_to_gt_boxes.values()] ), # number of ground truth objects
            Y_pred = Y_pred,
            Y_true = Y_true,
        )
        # Run
        froc_output_data = rafroc(froc_input_data, weights, matching_gt_idxs, gt_weights) # TODO
        # Save
        with open(output_dir / "rafroc_output_data.pkl", "wb") as f:
            pickle.dump(froc_output_data, f)
        with open(output_dir / "rafroc_score.json", "w") as f:
            json.dump({
                "score": froc_output_data.score,
            }, f, indent=4)
        froc_output_data.fig.savefig(output_dir / "rafroc.png")
        print(f">> WARNING: You are using example data. Expect weirdness in behavior (e.g. large changes in result) compared to real data. ") if sorted(set(mod_data.cases)) == ["Patient1", "Patient2", "Patient3", "Patient4", "Patient5", "Patient6", "Patient7", "Patient8"] else None
        print(f">> raFROC score: {froc_output_data.score}")
        print(f">> Done. Results saved at {str(output_dir.resolve())}")


if __name__ == "__main__":
    main()