from __future__ import print_function

import argparse
import os
import shutil

def parse_args():
    # Construct argument parser.
    parser = argparse.ArgumentParser(description = "Utility to extract frames from videos.")
    parser.add_argument("--filename", type = str, help = "Video filename.", required = True)
    parser.add_argument("--mode", type = str, help = "Preview or final.", default = "preview")
    parser.add_argument("--input_path", type = str, help = "Input directory.")
    parser.add_argument("--working_path", type = str, help = "Working directory.")
    parser.add_argument("--output_path", type = str, help = "Output directory.")
    parser.add_argument("--ffmpeg", help = "FFMPEG path.", default = "")
    parser.add_argument("--framerate", help = "Input video framerate.", default = "30000/1001")
    parser.add_argument("--shift", type = int, help = "Circular shift top or bottom video (when positive and negative respectively) by the number of input pixels.", default = 0)
    parser.add_argument("--sync", type = int, help = "Stereo time offset.", default = 0)
    parser.add_argument("--trim", type = int, help = "Number of frames to trim from start and end.", default = 0)
    parser.add_argument("--step", type = int, help = "Difference in frame number between consecutive frames.", default = 1)

    arguments = parser.parse_args()
    name = os.path.splitext(arguments.filename)[0]

    return arguments, name

def top_filter(arguments):
    if arguments.shift > 0:
        shift = abs(arguments.shift)
        return " -filter_complex \"[0:v][0:v]overlay={0}:0[bg];[bg][0:v]overlay={0}-W\"".format(shift)
    else:
        return ""
        
def bottom_filter(arguments):
    if arguments.shift < 0:
        shift = abs(arguments.shift)
        return " -filter_complex \"[0:v]hflip[tmp];[tmp]vflip[flip];[flip]split[f1][f2];[f1]split[f11][f12];[f11][f12]overlay={0}:0[bg];[bg][f2]overlay={0}-W\"".format(shift)
    else:
        return " -vf \"hflip,vflip\""

def preview(arguments, name):
    preview_path = os.path.join(arguments.working_path, name)

    # Check and create output directory for preview.
    if not os.path.exists(preview_path):
        os.makedirs(preview_path)
    
    # Extract frames using ffmpeg."
    preview_command = "{} -r {} -i {} -t 7 -qscale:v 2 {}"
    
    top_preview_command = preview_command.format(
        os.path.join(arguments.ffmpeg, "ffmpeg"),
        arguments.framerate,
        os.path.join(arguments.input_path, "top", arguments.filename) + top_filter(arguments),
        os.path.join(preview_path, "frame_%06dt.jpg")
    )
    os.system(top_preview_command)
    
    bottom_preview_command = preview_command.format(
        os.path.join(arguments.ffmpeg, "ffmpeg"),
        arguments.framerate,
        os.path.join(arguments.input_path, "bottom", arguments.filename) + bottom_filter(arguments),
        os.path.join(preview_path, "frame_%06db.jpg")
    )
    os.system(bottom_preview_command)

def extract_frames(arguments, name, folder = ""):
    offset = arguments.sync
    working_path = os.path.join(arguments.working_path, "tmp")

    # Extract all frames.
    if not os.path.exists(working_path):
        os.makedirs(working_path)

    extract_command = "{} -r {} -i {} -qscale:v 2 {}"
    os.system(
        extract_command.format(
            os.path.join(arguments.ffmpeg, "ffmpeg"),
            arguments.framerate,
            os.path.join(arguments.input_path, "top", arguments.filename),
            os.path.join(working_path, "frame_%06dt.jpg")
        )
    )
    os.system(
        extract_command.format(
            os.path.join(arguments.ffmpeg, "ffmpeg"),
            arguments.framerate,
            os.path.join(arguments.input_path, "bottom", arguments.filename),
            "-vf \"hflip,vflip\" " + os.path.join(working_path, "frame_%06db.jpg")
        )
    )
    
    # Rename and delete redundant frames.
    all_filenames = os.listdir(working_path)
    top_filenames = [filename for filename in all_filenames if filename.endswith("t.jpg")]
    bottom_filenames = [filename for filename in all_filenames if filename.endswith("b.jpg")]
    if offset > 0:
        top_index = offset + arguments.trim
        bottom_index = arguments.trim
    else:
        top_index = arguments.trim
        bottom_index = arguments.trim - offset
    
    index = 0
    if not os.path.exists(os.path.join(arguments.output_path, "top", folder, name)):
        os.makedirs(os.path.join(arguments.output_path, "top", folder, name))
    if not os.path.exists(os.path.join(arguments.output_path, "bottom", folder, name)):
        os.makedirs(os.path.join(arguments.output_path, "bottom", folder, name))
    
    while top_index < (len(top_filenames) - arguments.trim) and bottom_index < (len(bottom_filenames) - arguments.trim):
        src_top = os.path.join(working_path, top_filenames[top_index])
        src_bottom = os.path.join(working_path, bottom_filenames[bottom_index])
        dst_top = os.path.join(arguments.output_path, "top", folder, name, "{:06}.jpg".format(index))
        dst_bottom = os.path.join(arguments.output_path, "bottom", folder, name, "{:06}.jpg".format(index))
        shutil.copy(src_top, dst_top)
        shutil.copy(src_bottom, dst_bottom)
        top_index += arguments.step
        bottom_index += arguments.step
        index += 1
    
    # Delete temporary directory.
    shutil.rmtree(working_path)

if __name__ == "__main__":
    arguments, name = parse_args()
    if arguments.mode == "final":
        extract_frames(arguments, name)
    else:
        preview(arguments, name)