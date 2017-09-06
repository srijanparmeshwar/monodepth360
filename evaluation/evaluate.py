from __future__ import print_function

import argparse
import matplotlib.image as mpimg
import numpy as np
import os

from reader import read_file

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
    parser.add_argument("--filter_format", type = str, help = "Filter dark images.", default = "{}_top.jpg")
    parser.add_argument("--crop", type = int, help = "Crop test images vertically.", default = 0)

    arguments = parser.parse_args()

    return arguments

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

# Search for corresponding RGB images in nearby folders.
def search(start_index):
    if arguments.filter_format.format(start_index) in os.listdir(arguments.predicted_path):
        return arguments.predicted_path
    else:
        parent_directory = os.path.dirname(arguments.predicted_path)
        directories = [os.path.join(parent_directory, filename)
                       for filename
                       in os.listdir(parent_directory)
                       if os.path.isdir(os.path.join(parent_directory, filename))]
        for directory in directories:
            if arguments.filter_format.format(start_index) in os.listdir(directory):
                return directory

        print("Could not find RGB images for filtering. Exiting.")
        exit(1)

# Filter dark images and ones where walls are too close to the camera.
def filter_bad_images():
    index = 0
    gt_index = arguments.gt_start
    predicted_index = arguments.predicted_start
    indices = []
    filter_path = search(predicted_index)

    filter_filename = os.path.join(arguments.gt_path, "filter_close.txt")
    if os.path.exists(filter_filename):
        with open(filter_filename, "r") as filter_file:
            filter_close_indices = [int(line.strip()) for line in filter_file.readlines()]
    else:
        filter_close_indices = []

    for image_index in range(arguments.samples):
        rgb = mpimg.imread(os.path.join(filter_path, arguments.filter_format.format(predicted_index)))
        if np.median(rgb) > 5 and gt_index not in filter_close_indices:
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

    # Iterate over predicted and ground truth examples.
    for index, gt_index, predicted_index in indices:
        print("Evaluating image {}.".format(index))

        # Check for baseline median value in directory.
        baseline = None
        for filename in os.listdir(arguments.predicted_path):
            if filename.endswith(".txt"):
                baseline = os.path.join(arguments.predicted_path, filename)

        # If file exists, evaluate as baseline value.
        if baseline is not None:
            with open(baseline, "r") as file:
                predicted_value = float(file.read())
            ground_truth, mask = read_file(os.path.join(arguments.gt_path, arguments.gt_format.format(gt_index)))
            predicted = np.full(ground_truth.shape, predicted_value)
        else:
            predicted, _ = read_file(os.path.join(arguments.predicted_path, arguments.predicted_format.format(predicted_index)))
            ground_truth, mask = read_file(os.path.join(arguments.gt_path, arguments.gt_format.format(gt_index)),
                                           predicted.shape)

        predicted *= arguments.scale

        # Clip values to be between min_depth and max_depth.
        predicted[predicted < arguments.min_depth] = arguments.min_depth
        predicted[predicted > arguments.max_depth] = arguments.max_depth

        ground_truth[ground_truth < arguments.min_depth] = arguments.min_depth
        ground_truth[ground_truth > arguments.max_depth] = arguments.max_depth

        # Crop images vertically if requested.
        if arguments.crop > 0:
            predicted = predicted[arguments.crop:-arguments.crop, :]
            ground_truth = ground_truth[arguments.crop:-arguments.crop, :]
            mask = mask[arguments.crop:-arguments.crop, :]

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

    # Calculate mean values across dataset.
    metrics = [
        "ABS: {:.4f}".format(abs_rel.mean()),
        "SQ: {:.4f}".format(sq_rel.mean()),
        "RMS: {:.4f}".format(rms.mean()),
        "RMSlog: {:.4f}".format(log_rms.mean()),
        "A1: {:.4f}".format(a1.mean()),
        "A2: {:.4f}".format(a2.mean()),
        "A3: {:.4f}".format(a3.mean())
    ]

    # Print results to console.
    print("\n".join(["---Metrics---"] + metrics))
    print("\n".join(["---LaTeX---", " & ".join([str(float(x.split(":")[1])) for x in metrics])]))

    if arguments.crop > 0:
        results_filename = os.path.split(arguments.predicted_path)[-1] + "_{}.txt".format(arguments.crop)
    else:
        results_filename = os.path.split(arguments.predicted_path)[-1] + ".txt"

    # Write results to file.
    with open(results_filename, "w") as results_file:
        results_file.write("\n".join(["---Metrics---"] + metrics))

if __name__ == "__main__":
    arguments = parse_args()
    evaluate()
