import argparse
import numpy as np
import os
import shutil
import tensorflow as tf

from argparse import Namespace
from convert import e2c
from image_utils import *
from spherical import equirectangular_to_cubic

def parse_args():
    # Construct argument parser.
    parser = argparse.ArgumentParser(description = 'Calibration utility.')
    parser.add_argument("--filename", type = str, help = "Video filename.", required = True)
    parser.add_argument("--input_path", type = str, help = "Path to directory containing input videos.")
    parser.add_argument("--working_path", type = str, help = "Path to output directory.")
    parser.add_argument("--output_path", type = str, help = "Output directory.")
    parser.add_argument("--input_height", type = int, help = "Input height.", default = 2048)
    parser.add_argument("--input_width", type = int, help = "Input width.", default = 4096)
    parser.add_argument("--output_height", type = int, help = "Output height.", default = 1024)
    parser.add_argument("--output_width", type = int, help = "Output width.", default = 1024)
    parser.add_argument("--batch_size", type = int, help = "Batch size for TensorFlow processing.", default = 16)
    parser.add_argument("--recompute", help = "Recompute SFM.", action = "store_true")
    parser.add_argument("--ffmpeg", help = "FFMPEG path.", default = "")
    parser.add_argument("--sfmrecon", help = "MVE sfmrecon path.", default = "")
    parser.add_argument("--makescene", help = "MVE makescene path.", default = "")
    parser.add_argument("--framerate", help = "Input video framerate.", default = "30000/1001")
    parser.add_argument("--t_offset", type = int, help = "Stereo time offset.", default = 0)
    parser.add_argument("--n_offset", type = int, help = "Filename offset.", default = 0)

    arguments = parser.parse_args()
    fraction = [float(element) for element in arguments.framerate.split("/")]
    framerate = fraction[0] / fraction[1]
    return arguments, framerate

def vid2seq():
    # Check and create output directory for frames.
    if not os.path.exists(os.path.join(arguments.working_path, "calib")):
        os.makedirs(os.path.join(arguments.working_path, "calib"))
    
    # Extract frames using ffmpeg."
    os.system(os.path.join(arguments.ffmpeg, "ffmpeg") + " -ss 00:00:05 -r " + arguments.framerate + " -i " + os.path.join(arguments.input_path, "top", arguments.filename) + " -t 5 -qscale:v 2 " + os.path.join(arguments.working_path, "calib", "image_%06dt.jpg"))
    os.system(os.path.join(arguments.ffmpeg, "ffmpeg") + " -ss 00:00:05 -r " + arguments.framerate + " -i " + os.path.join(arguments.input_path, "bottom", arguments.filename) + " -t 5 -qscale:v 2 -vf \"hflip,vflip\" " + os.path.join(arguments.working_path, "calib", "image_%06db.jpg"))
    
def seq2face():
    # Check and create output directory for cubic images.
    if not os.path.exists(os.path.join(arguments.working_path, "calib", "sfm")):
        os.makedirs(os.path.join(arguments.working_path, "calib", "sfm"))
    
    # Run equirectangular to cubic converter to extract left side (index 2).
    convert_arguments = Namespace(
        input_path = os.path.join(arguments.working_path, "calib"),
        working_path = os.path.join(arguments.working_path, "calib", "sfm"),
        input_format = "jpg",
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
    if not os.path.exists(os.path.join(arguments.working_path, "calib", "sfm", "views")):
        os.system(os.path.join(arguments.makescene, "makescene") + " -i " + os.path.join(arguments.working_path, "calib", "sfm") + " " + os.path.join(arguments.working_path, "calib", "sfm"))
    if os.path.exists(os.path.join(arguments.working_path, "calib", "sfm", "prebundle.sfm")) and not arguments.recompute:
        os.system(os.path.join(arguments.sfmrecon, "sfmrecon") + " --initial-pair=100,101 --prebundle=" + os.path.join(arguments.working_path, "calib", "sfm", "prebundle.sfm") + " --shared-intrinsics --video-matching=20 " + os.path.join(arguments.working_path, "calib", "sfm"))
    else:
        os.system(os.path.join(arguments.sfmrecon, "sfmrecon") + " --initial-pair=100,101 --shared-intrinsics --video-matching=20 " + os.path.join(arguments.working_path, "calib", "sfm"))
    
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
        
        offset = int((self.top_indices[amin_t[0]] + self.top_indices[amin_b[0]] - self.bottom_indices[amin_t[1]] - self.bottom_indices[amin_b[1]]) / 2.0)
        y_distance = abs(y_t[amin_t[0], amin_t[1], 1] + y_b[amin_t[0], amin_t[1], 1]) / 2.0
        with open(os.path.join(arguments.working_path, "calib.txt"), "w") as calib_file:
            calib_file.write(str(y_distance[0]))

        print(offset)
        return offset
        
def ncc(x, Y, batch_size):
    xb = tf.reduce_mean(x, [1, 2], keep_dims=True)
    Yb = tf.reduce_mean(Y, [1, 2], keep_dims=True)
    xv = tf.reduce_mean((x - xb) ** 2.0, [1, 2], keep_dims=True)
    Yv = tf.reduce_mean((Y - Yb) ** 2.0, [1, 2], keep_dims=True)
    
    return tf.reduce_sum((x - xb) * (Y - Yb) / tf.sqrt(xv * Yv), [1, 2, 3])
    
def f(tfs, bfs, tids, bids):
    ds = []
    idx = []
    for i in range(len(tfs)):
        topf = tfs[i]
        tid = tids[i]
        for j in range(len(bfs)):
            bf = bfs[j]
            bid = bids[j]
            with tf.Graph().as_default(), tf.Session() as session:
                top = read_image(topf, [64, 128])
                bottom = read_image(bf, [64, 128])
                faces_t = equirectangular_to_cubic(top, [32, 32])
                faces_b = equirectangular_to_cubic(bottom, [32, 32])
                d = session.run(ncc(faces_t[0], faces_b[0], 1))
                print("({}, {}): {}".format(tid, bid, d))
                ds.append(d.tolist())
                idx.append((tid, bid))
    
    id = np.argmax(np.array(ds))
    return idx[id]
    
def g(t, b, tids, bids):
    return [t[i] for i in tids], [b[i] for i in bids]

def find_offset(path):
    all_filenames = os.listdir(os.path.join(arguments.working_path, "tmp"))
    top_filenames = [os.path.join(arguments.working_path, "tmp", filename) for filename in all_filenames if filename.endswith("t.jpg")]
    bottom_filenames = [os.path.join(arguments.working_path, "tmp", filename) for filename in all_filenames if filename.endswith("b.jpg")]
    tids = range(75, 125)
    bids = range(0, 200, 10)
    tfs, bfs = g(top_filenames, bottom_filenames, tids, bids)
    tid, bid = f(tfs, bfs, tids, bids)
    bids = range(max(0, bid - 10), min(200, bid + 10))
    tfs, bfs = g(top_filenames, bottom_filenames, tids, bids)
    ids = f(tfs, bfs, tids, bids)
    print(ids)

    # for topf in top_filenames:
        # ds = []
        # image_index = 0
        # while image_index < len(bottom_filenames):
            # bfs = bottom_filenames[image_index:min(image_index + arguments.batch_size, len(bottom_filenames))]
            # with tf.Graph().as_default(), tf.Session() as session:
                # top = read_image(topf, [32, 32])
                # bs = tf.concat([read_image(filename, [32, 32]) for filename in bfs], 0)
                # d = session.run(ncc(top, bs, min(arguments.batch_size, len(bfs))))
                # ds.extend(d.tolist())
            
            # image_index += arguments.batch_size
        # print(ds)
        # print(np.argmax(np.array(ds)))
    
    

    idx = []
    ds = []
    i = 100
    for topf in top_filenames:
        j = 0
        for bf in bottom_filenames:
            with tf.Graph().as_default(), tf.Session() as session:
                with tf.device("/cpu:0"):
                    top = read_image(topf, [64, 128])
                    bottom = read_image(bf, [64, 128])
                    faces_t = equirectangular_to_cubic(top, [32, 32])
                    faces_b = equirectangular_to_cubic(bottom, [32, 32])
                    d = session.run(ncc(faces_t[0], faces_b[0], 1))
                    print("({}, {}): {}".format(i, j, d))
                    ds.append(d.tolist())
                    idx.append((i, j))
            j += 10
        i += 1
    
    print(np.argmax(np.array(ds)))
    print(idx[np.argmax(np.array(ds))])
    
def get_last_index(path):
    filenames = os.listdir(path)
    indices = [int(os.path.splitext(os.path.basename(filename))[0]) for filename in filenames]
    return sorted(indices)[-1]

def calibrate():
    # Find calibration parameters.
    # calibrator = Calibrator(os.path.join(arguments.working_path, "calib", "sfm", "synth_0.out"))
    # offset = calibrator.optimise()
    offset = arguments.t_offset
    
    # Extract all frames.
    if not os.path.exists(os.path.join(arguments.working_path, "tmp")):
        os.makedirs(os.path.join(arguments.working_path, "tmp"))
    
    os.system(os.path.join(arguments.ffmpeg, "ffmpeg") + " -r " + str(framerate) + " -i " + os.path.join(arguments.input_path, "top", arguments.filename) + " -qscale:v 2 " + os.path.join(arguments.working_path, "tmp", "image_%03dt.jpg"))
    os.system(os.path.join(arguments.ffmpeg, "ffmpeg") + " -r " + str(framerate) + " -i " + os.path.join(arguments.input_path, "bottom", arguments.filename) + " -qscale:v 2 -vf \"hflip,vflip\" " + os.path.join(arguments.working_path, "tmp", "image_%03db.jpg"))
    
    # Rename and delete redundant frames.
    all_filenames = os.listdir(os.path.join(arguments.working_path, "tmp"))
    top_filenames = [filename for filename in all_filenames if filename.endswith("t.jpg")]
    bottom_filenames = [filename for filename in all_filenames if filename.endswith("b.jpg")]
    if offset > 0:
        top_index = offset
        bottom_index = 0
    else:
        top_index = 0
        bottom_index = - offset
    
    index = 0
    if not os.path.exists(os.path.join(arguments.output_path, "top")):
        os.makedirs(os.path.join(arguments.output_path, "top"))
    else:
        arguments.n_offset = get_last_index(os.path.join(arguments.output_path, "top")) + 1
    if not os.path.exists(os.path.join(arguments.output_path, "bottom")):
        os.makedirs(os.path.join(arguments.output_path, "bottom"))
    
    while top_index < len(top_filenames) and bottom_index < len(bottom_filenames):
        src_top = os.path.join(arguments.working_path, "tmp", top_filenames[top_index])
        src_bottom = os.path.join(arguments.working_path, "tmp", bottom_filenames[bottom_index])
        dst_top = os.path.join(arguments.output_path, "top", "{}.jpg".format(index + arguments.n_offset))
        dst_bottom = os.path.join(arguments.output_path, "bottom", "{}.jpg".format(index + arguments.n_offset))
        shutil.copy(src_top, dst_top)
        shutil.copy(src_bottom, dst_bottom)
        top_index += 1
        bottom_index += 1
        index += 1
    
    # Delete temporary directory.
    shutil.rmtree(os.path.join(arguments.working_path, "tmp"))

if __name__ == "__main__":
    arguments, framerate = parse_args()
    #vid2seq()
    #seq2face()
    #sfm()
    calibrate()


