import matplotlib.image as mpimg
import numpy as np

from exr import read_depth
from scipy.ndimage import zoom

def read_file(filename, shape = None):
    if filename.lower().endswith(".exr"):
        depth_map = read_depth(filename)
        return depth_map, depth_map < 1000.0

    elif filename.lower().endswith(".png"):
        depth_map = mpimg.imread(filename)

        if shape is not None:
            ih, iw = depth_map.shape
            h, w = shape

            if ih > 1024:
                depth_map = depth_map[::2, ::2]

            depth_map = zoom(depth_map, [float(h) / float(ih), w / float(iw)], order = 1)

        mask = depth_map < 0.99
        depth_map = depth_map * 65536 / 1000
        return depth_map, mask

    elif filename.lower().endswith(".npy"):
        return np.load(filename), None