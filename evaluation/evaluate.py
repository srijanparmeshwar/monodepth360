import argparse
import numpy as np
import os

from exr import read_exr_depth

# Based on code from C. Godard's Monodepth evaluation code at https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
# Adapted to work directly on depth maps rather than converting from disparity maps.

def parse_args():
    parser = argparse.ArgumentParser(description = "Evaluate depth estimation.")
    parser.add_argument("--gt_path", type = str, help = "Path to ground truth data.", required = True)
    parser.add_argument("--predicted_path", type = str, help = "Path to predicted data.", required = True)
    parser.add_argument("--gt_format", type = str, help = "Format of ground truth filenames.", required = True)
    parser.add_argument("--predicted_format", type = str, help = "Format of predicted filenames.", default = "{}_depth.npy")
    parser.add_argument("--gt_start", type = int, help = "Start index for ground truth data.", default = 0)
    parser.add_argument("--predicted_start", type = int, help = "Start index for predicted data.", default = 0)
    parser.add_argument("--samples", type = int, help = "Number of samples to evaluate.", required = True)
    parser.add_argument("--min_depth", type = float, help = "Minimum depth for evaluation", default = 1e-3)
    parser.add_argument("--max_depth", type= float, help = "Maximum depth for evaluation", default = 100.0)
    parser.add_argument("--scale", type = float, help = "Scale factor for predicted depth maps.", default = 1.0)

    arguments = parser.parse_args()

    return arguments

def read_file(filename):
    if filename.lower().endswith(".exr"):
        return read_exr_depth(filename)
    elif filename.lower().endswith(".npy"):
        return np.load(filename)

def compute_errors(ground_truth, predicted, scale_factor = 1.0):
    predicted *= scale_factor

    thresh = np.maximum((ground_truth / predicted), (predicted / ground_truth))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2.0).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (ground_truth - predicted) ** 2.0
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(ground_truth) - np.log(predicted)) ** 2.0
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(ground_truth - predicted) / ground_truth)

    sq_rel = np.mean(((ground_truth - predicted) ** 2.0) / ground_truth)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def evaluate():
    rms     = np.zeros(arguments.samples, np.float32)
    log_rms = np.zeros(arguments.samples, np.float32)
    abs_rel = np.zeros(arguments.samples, np.float32)
    sq_rel  = np.zeros(arguments.samples, np.float32)
    a1      = np.zeros(arguments.samples, np.float32)
    a2      = np.zeros(arguments.samples, np.float32)
    a3      = np.zeros(arguments.samples, np.float32)

    gt_index = arguments.gt_start
    predicted_index = arguments.predicted_start

    for index in range(arguments.samples):
        ground_truth = read_file(os.path.join(arguments.gt_path, arguments.gt_format.format(gt_index)))
        predicted = read_file(os.path.join(arguments.predicted_path, arguments.predicted_format.format(predicted_index)))

        predicted[predicted < arguments.min_depth] = arguments.min_depth
        predicted[predicted > arguments.max_depth] = arguments.max_depth

        ground_truth[ground_truth < arguments.min_depth] = arguments.min_depth
        ground_truth[ground_truth > arguments.max_depth] = arguments.max_depth

        abs_rel[index], sq_rel[index], rms[index], log_rms[index], a1[index], a2[index], a3[index] = compute_errors(ground_truth,
                                                                                        predicted,
                                                                                        arguments.scale)

        gt_index += 1
        predicted_index += 1

    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(),
                                                                                                  sq_rel.mean(),
                                                                                                  rms.mean(),
                                                                                                  log_rms.mean(),
                                                                                                  a1.mean(),
                                                                                                  a2.mean(),
                                                                                                  a3.mean()))

if __name__ == "__main__":
    arguments = parse_args()
    evaluate()
