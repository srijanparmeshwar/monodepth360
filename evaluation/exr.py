import Imath
import numpy as np
import OpenEXR

FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

def read_depth(filename):
    file = OpenEXR.InputFile(filename)
    data_window = file.header()["dataWindow"]
    size = (data_window.max.x - data_window.min.x + 1, data_window.max.y - data_window.min.y + 1)
    z_string = file.channel("R", FLOAT)
    z = np.fromstring(z_string, dtype = np.float32)
    z.shape = (size[1], size[0])
    return z