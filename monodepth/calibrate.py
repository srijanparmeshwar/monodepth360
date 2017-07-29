import argparse
import numpy as np
import os
import tensorflow as tf

from argparse import Namespace
from convert import e2c

def parse_args():
    # Construct argument parser.
    parser = argparse.ArgumentParser(description = 'Calibration utility.')
    parser.add_argument("--filename", type = str, help = "Video filename.", required = True)
    parser.add_argument("--input_path", type = str, help = "Path to directory containing input videos.")
    parser.add_argument("--output_path", type = str, help = "Path to output directory.")
    parser.add_argument("--input_height", type = int, help = "Input height.", default = 2048)
    parser.add_argument("--input_width", type = int, help = "Input width.", default = 4096)
    parser.add_argument("--output_height", type = int, help = "Output height.", default = 1024)
    parser.add_argument("--output_width", type = int, help = "Output width.", default = 1024)
    parser.add_argument("--batch_size", type = int, help = "Batch size for TensorFlow processing.", default = 8)
    parser.add_argument("--recompute", help = "Recompute SFM.", action = "store_true")
    parser.add_argument("--sfmrecon", help = "MVE sfmrecon path.", default = "")
    parser.add_argument("--makescene", help = "MVE makescene path.", default = "")

    return parser.parse_args()

def vid2seq():
    # Check and create output directory for frames.
    if not os.path.exists(os.path.join(arguments.output_path, "calib")):
        os.makedirs(os.path.join(arguments.output_path, "calib"))
    
    # Extract frames using ffmpeg."
    os.system("ffmpeg -r 15000/1001 -ss 00:00:05 -i " + os.path.join(arguments.input_path, "top", arguments.filename) + " -t 7.5 " + os.path.join(arguments.output_path, "calib", "image_%03dt.png"))
    os.system("ffmpeg -r 15000/1001 -ss 00:00:05 -i " + os.path.join(arguments.input_path, "bottom", arguments.filename) + " -t 7.5 -vf \"hflip,vflip\" " + os.path.join(arguments.output_path, "calib", "image_%03db.png"))
    
def seq2face():
    # Check and create output directory for cubic images.
    if not os.path.exists(os.path.join(arguments.output_path, "calib", "sfm")):
        os.makedirs(os.path.join(arguments.output_path, "calib", "sfm"))
    
    # Run equirectangular to cubic converter to extract left side (index 2).
    convert_arguments = Namespace(
        input_path = os.path.join(arguments.output_path, "calib"),
        output_path = os.path.join(arguments.output_path, "calib", "sfm"),
        input_format = "png",
        output_format = "jpg",
        input_height = arguments.input_height,
        input_width = arguments.input_width,
        output_height = arguments.output_height,
        output_width = arguments.output_width,
        faces = "2",
        batch_size = arguments.batch_size
    )
    e2c(convert_arguments)
    
def sfm():
    # Run MVE structure from motion.
    # Stores result in <output_directory>/calib/sfm/synth_0.out
    if not os.path.exists(os.path.join(arguments.output_path, "calib", "sfm", "views")):
        os.system(os.path.join(arguments.makescene, "makescene") + " -i " + os.path.join(arguments.output_path, "calib", "sfm") + " " + os.path.join(arguments.output_path, "calib", "sfm"))
    if os.path.exists(os.path.join(arguments.output_path, "calib", "sfm", "prebundle.sfm")) and not arguments.recompute:
        os.system(os.path.join(arguments.sfmrecon, "sfmrecon") + " --prebundle=" + os.path.join(arguments.output_path, "calib", "sfm", "prebundle.sfm") + " --shared-intrinsics --video-matching=30 " + os.path.join(arguments.output_path, "calib", "sfm"))
    else:
        os.system(os.path.join(arguments.sfmrecon, "sfmrecon") + " --shared-intrinsics --video-matching=30 " + os.path.join(arguments.output_path, "calib", "sfm"))
    
class Calibrator:

    def __init__(self, filename):
        # Load SFM output.
        self.parse_file(filename)
        
    def parse_file(self, filename):
        with open(filename, "r") as file:
            lines = file.readlines()
        
        self.top_indices = []
        self.bottom_indices = []
        
        n_cameras = int(lines[1].split(" ")[0])
        top_rotations = []
        top_translations = []
        bottom_rotations = []
        bottom_translations = []
        
        line_index = 2
        for camera_index in range(n_cameras):
            # Check if focal length and process if not zero.
            if not float(lines[line_index].split(" ")[0]) == 0.0:
                # Parse rotations and translations for each camera view.
                py_rotations = [[float(element) for element in lines[index].split(" ")] for index in [line_index + 1, line_index + 2, line_index + 3]]
                py_translations = [float(element) for element in lines[line_index + 4].split(" ")]
                # Convert to Numpy arrays and add to appropriate lists.
                if camera_index % 2 == 0:
                    bottom_rotations.append(np.transpose(np.stack(py_rotations)))
                    bottom_translations.append(np.array(py_translations))
                    self.bottom_indices.append(camera_index / 2)
                else:
                    top_rotations.append(np.transpose(np.stack(py_rotations)))
                    top_translations.append(np.array(py_translations))
                    self.top_indices.append((camera_index - 1) / 2)
            
            line_index += 5
        
        # Stack lists into single N-D arrays.
        self.top_rotations = np.stack(top_rotations)
        self.top_translations = np.stack(top_translations)
        self.bottom_rotations = np.stack(bottom_rotations)
        self.bottom_translations = np.stack(bottom_translations)
    
    def repeat(self, x, axis, n):
        if axis == 0:
            return np.tile(np.expand_dims(np.expand_dims(x, 0), 3), [n, 1, 1, 1])
        else:
            return np.tile(np.expand_dims(np.expand_dims(x, 1), 3), [1, n, 1, 1])
        
    def optimise(self):
        m = self.top_rotations.shape[0]
        n = self.bottom_rotations.shape[0]
        
        # Pairwise displacements between computed camera centres.
        x_t = self.repeat(self.top_translations, 1, n)
        x_b = self.repeat(self.bottom_translations, 0, m)
        x = x_t - x_b
        
        # Apply rotations to displacements using respective rotation matrices.
        y_t = np.einsum("ikl,ijl...->ijk...", self.top_rotations, x)
        y_b = np.einsum("jkl,ijl...->ijk...", self.bottom_rotations, x)
        
        # Calculate XZ distances.
        d_t = y_t[:, :, 0, :] ** 2.0 + y_t[:, :, 2, :] ** 2.0
        d_b = y_b[:, :, 0, :] ** 2.0 + y_b[:, :, 2, :] ** 2.0
        
        # Compute minimum arguments.
        amin_t = np.unravel_index(np.argmin(d_t), d_t.shape)
        amin_b = np.unravel_index(np.argmin(d_b), d_b.shape)
        
        # Print results.
        print("Top (i, j): " + str(self.top_indices[amin_t[0]]) + ", " + str(self.bottom_indices[amin_t[1]]))
        print("Bottom (i, j): " + str(self.top_indices[amin_b[0]]) + ", " + str(self.bottom_indices[amin_b[1]]))
        print("Top Y-distance: " + str(y_t[amin_t[0], amin_t[1], 1]))
        print("Bottom Y-distance: " + str(y_b[amin_t[0], amin_t[1], 1]))
        
def calibrate():
    calibrator = Calibrator(os.path.join(arguments.output_path, "calib", "sfm", "synth_0.out"))
    calibrator.optimise()

if __name__ == "__main__":
    arguments = parse_args()
    vid2seq()
    seq2face()
    sfm()
    calibrate()


