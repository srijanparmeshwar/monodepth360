import argparse
import os
import tensorflow as tf

from argparse import Namespace
from convert import e2c

# Construct argument parser.
parser = argparse.ArgumentParser(description = 'Calibration utility.')
parser.add_argument("--filename", type = str, help = "Video filename.", required = True)
parser.add_argument("--input_path", type = str, help = "Path to directory containing input videos.")
parser.add_argument("--output_path", type = str, help = "Path to output directory.")
parser.add_argument("--input_height", type = int, help = "Input height.", default = 2048)
parser.add_argument("--input_width", type = int, help = "Input width.", default = 4096)
parser.add_argument("--output_height", type = int, help = "Output height.", default = 1024)
parser.add_argument("--output_width", type = int, help = "Output width.", default = 1024)

arguments = parser.parse_args()

def vid2seq():
    # Check and create output directory for frames.
    if not os.path.exists(os.path.join(arguments.output_path, "calib")):
        os.makedirs(os.path.join(arguments.output_path, "calib"))
    
    os.system("ffmpeg -ss 00:00:05 -i " + os.path.join(arguments.input_path, "top", arguments.filename) + " -r 30000/1001 -t 3 " + os.path.join(arguments.output_path, "calib", "image_%03dt.png"))
    os.system("ffmpeg -ss 00:00:06 -i " + os.path.join(arguments.input_path, "bottom", arguments.filename) + " -r 30000/1001 -t 3 -vf \"hflip,vflip\" " + os.path.join(arguments.output_path, "calib", "image_%03db.png"))
    
def seq2face():
    # Check and create output directory for cubic images.
    if not os.path.exists(os.path.join(arguments.output_path, "calib", "sfm")):
        os.makedirs(os.path.join(arguments.output_path, "calib", "sfm"))
    
    # Run equirectangular to cubic converter to extract left side.
    convert_arguments = Namespace(
        input_path = os.path.join(arguments.output_path, "calib"),
        output_path = os.path.join(arguments.output_path, "calib", "sfm"),
        input_format = "png",
        output_format = "jpg",
        input_height = arguments.input_height,
        input_width = arguments.input_width,
        output_height = arguments.output_height,
        output_width = arguments.output_width,
        faces = "2"
    )
    e2c(convert_arguments)
    
def sfm():
    # Run VisualSFM with shared and fixed intrinsic calibration.
    # Stores result in <output_directory>/calib/sfm/result.nvm
    os.system("makescene -i " + os.path.join(arguments.output_path, "calib", "sfm") + " " + os.path.join(arguments.output_path, "calib", "sfm"))
    #os.system("VisualSFM sfm+shared+sort+k=512,512,512,512 " + os.path.join(arguments.output_path, "calib", "sfm/") + " " + os.path.join(arguments.output_path, "calib", "sfm", "result.nvm"))
    os.system("sfmrecon --shared-intrinsics " + os.path.join(arguments.output_path, "calib", "sfm"))

if __name__ == "__main__":
    #vid2seq()
    #seq2face()
    sfm()


