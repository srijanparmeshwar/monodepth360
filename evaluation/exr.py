import array
import Imath
import numpy as np
import OpenEXR

FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

def read_exr_depth(filename):
    file = OpenEXR.InputFile(filename)
    data_window = file.header()['dataWindow']
    size = (data_window.max.x - data_window.min.x + 1, data_window.max.y - data_window.min.y + 1)
    array_Z = array.array('f', file.channel("R", FLOAT))
    Z = np.transpose(np.reshape(np.asarray(array_Z), size), [1, 0])
    return Z