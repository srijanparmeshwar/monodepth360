from __future__ import print_function

import argparse
import matplotlib.image as mpimg
import numpy as np
import os

from exr import read_depth
from scipy.ndimage import zoom

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
    parser.add_argument("--min_depth", type = float, help = "Minimum depth for evaluation.", default = 1e-3)
    parser.add_argument("--max_depth", type= float, help = "Maximum depth for evaluation.", default = 80.0)
    parser.add_argument("--scale", type = float, help = "Scale factor for predicted depth maps.", default = 1.0)
    parser.add_argument("--filter", help = "Filter dark images.", action = "store_true")

    arguments = parser.parse_args()

    return arguments

def read_file(filename, shape = None):
    if filename.lower().endswith(".exr"):
        return read_depth(filename), None
    elif filename.lower().endswith(".png"):
        depth_map = mpimg.imread(filename)
        if shape is not None:
            ih, iw = depth_map.shape
            h, w = shape
            depth_map = zoom(depth_map[::2, ::2], [float(h) / float(ih), w / float(iw)], order = 1)
        mask = depth_map < 0.99
        depth_map = depth_map * 65536 / 1000
        return depth_map, mask
    elif filename.lower().endswith(".npy"):
        return np.load(filename), None

def compute_errors(ground_truth, predicted):
    thresh = np.maximum((ground_truth / predicted), (predicted / ground_truth))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2.0).mean()
    a3 = (thresh < 1.25 ** 3.0).mean()

    rmse = (ground_truth - predicted) ** 2.0
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(ground_truth) - np.log(predicted)) ** 2.0
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(ground_truth - predicted) / ground_truth)

    sq_rel = np.mean(((ground_truth - predicted) ** 2.0) / ground_truth)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def filter_bad_images():
    index = 0
    gt_index = arguments.gt_start
    predicted_index = arguments.predicted_start
    indices = []
    for image_index in range(arguments.samples):
        rgb = mpimg.imread(os.path.join(arguments.predicted_path, "{}_top.jpg").format(predicted_index))
        if np.median(rgb) > 5:
            indices.append((index, gt_index, predicted_index))
            index += 1

        gt_index += 1
        predicted_index += 1

    return indices

def get_indices():
    if arguments.filter:
        return filter_bad_images()

    index = 0
    gt_index = arguments.gt_start
    predicted_index = arguments.predicted_start
    indices = []
    for image_index in range(arguments.samples):
        indices.append((index, gt_index, predicted_index))
        index += 1
        gt_index += 1
        predicted_index += 1

    return indices


def evaluate():
    indices = get_indices()
    samples = len(indices)
    print("Filtered {} examples. Evaluating the remaining {} examples.".format(arguments.samples - samples, samples))

    rms     = np.zeros(samples, np.float32)
    log_rms = np.zeros(samples, np.float32)
    abs_rel = np.zeros(samples, np.float32)
    sq_rel  = np.zeros(samples, np.float32)
    a1      = np.zeros(samples, np.float32)
    a2      = np.zeros(samples, np.float32)
    a3      = np.zeros(samples, np.float32)

    for index, gt_index, predicted_index in indices:
        print("Evaluating image {}.".format(index))
        predicted, _ = read_file(os.path.join(arguments.predicted_path, arguments.predicted_format.format(predicted_index)))
        predicted *= arguments.scale

        ground_truth, mask = read_file(os.path.join(arguments.gt_path, arguments.gt_format.format(gt_index)), predicted.shape)

        predicted[predicted < arguments.min_depth] = arguments.min_depth
        predicted[predicted > arguments.max_depth] = arguments.max_depth

        ground_truth[ground_truth < arguments.min_depth] = arguments.min_depth
        ground_truth[ground_truth > arguments.max_depth] = arguments.max_depth

        if mask is None:
            x = ground_truth
            y = predicted
        else:
            x = ground_truth[mask]
            y = predicted[mask]
        abs_rel[index], sq_rel[index], rms[index], log_rms[index], a1[index], a2[index], a3[index] = compute_errors(x, y)
        print("ABS: {:.4f}, SQ: {:.4f}, RMS: {:.4f}, logRMS: {:.4f}, A1: {:.4f}, A2: {:.4f}, A3: {:.4f}".format(
            abs_rel[index], sq_rel[index], rms[index], log_rms[index], a1[index], a2[index], a3[index]
        ))

    print("---Metrics---")
    print("ABS: {:.4f}".format(abs_rel.mean()))
    print("SQ: {:.4f}".format(sq_rel.mean()))
    print("RMS: {:.4f}".format(rms.mean()))
    print("logRMS: {:.4f}".format(log_rms.mean()))
    print("A1: {:.4f}".format(a1.mean()))
    print("A2: {:.4f}".format(a2.mean()))
    print("A3: {:.4f}".format(a3.mean()))

if __name__ == "__main__":
    arguments = parse_args()
    evaluate()
