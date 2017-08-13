import argparse
import os

from argparse import Namespace
from vid_to_seq import extract_frames
from vid_to_seq import preview

def parse_args():
    # Construct argument parser.
    parser = argparse.ArgumentParser(description = "Utility to extract frames from videos.")
    parser.add_argument("--mode", type = str, help = "Preview or final.", default = "preview")
    parser.add_argument("--input_path", type = str, help = "Input directory.")
    parser.add_argument("--working_path", type = str, help = "Working directory.")
    parser.add_argument("--output_path", type = str, help = "Output directory.")
    parser.add_argument("--ffmpeg", help = "FFMPEG path.", default = "")
    parser.add_argument("--framerate", help = "Input video framerate.", default = "30000/1001")
    parser.add_argument("--shift", type = int, help = "Circular shift top or bottom video (when positive and negative respectively) by the number of input pixels.", default = 0)
    parser.add_argument("--trim", type = int, help = "Number of frames to trim from start and end.", default = 0)
    parser.add_argument("--step", type = int, help = "Difference in frame number between consecutive frames.", default = 1)

    arguments = parser.parse_args()

    return arguments

scenes = {
    #"CanadaWater": [-92, 39, 10],
    #"CanaryWharf": [50, 204],
    "ChandlerHouse": [40]#, 71],
    #"FitzroySquare": [-36],
    #"FlaxmanGallery": [-57, 40, 45, 41],
    #"GreatPortlandStreet": [45, 40],
    #"HampsteadHeath": [34, 58, 37, 32, 38, 58],
    #"RussellSquare": [73],
    #"SenateHouse": [40],
    #"TottenhamCourtRoad": [40, 30],
    #"UCL": [-52, 51, 45, 41, 35]
    #"VictoriaHouse": [44, 36, 204]
}

def create_namespaces():
    namespaces = []
    names = []
    folders = []
    for scene_name in scenes.keys():
        scene = scenes[scene_name]
        for video_index in range(len(scene)):
            name = "{}".format(video_index + 1)
            namespace = Namespace(
                filename = "{}.MP4".format(video_index + 1),
                mode = arguments.mode,
                input_path = os.path.join(arguments.input_path, scene_name),
                output_path = arguments.output_path,
                working_path = os.path.join(arguments.working_path, scene_name),
                ffmpeg = arguments.ffmpeg,
                framerate = arguments.framerate,
                shift = arguments.shift,
                sync = scene[video_index],
                trim = arguments.trim,
                step = arguments.step
            )
            namespaces.append(namespace)
            names.append(name)
            folders.append(scene_name)

    return namespaces, names, folders

def process_scenes():
    namespaces, names, folders = create_namespaces()
    for index in range(len(namespaces)):
        namespace = namespaces[index]
        name = names[index]
        folder = folders[index]
        if arguments.mode == "final":
            extract_frames(namespace, name, folder)
        else:
            preview(namespace, name)

if __name__ == "__main__":
    arguments = parse_args()
    process_scenes()
