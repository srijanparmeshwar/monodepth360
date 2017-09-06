# monodepth360

Master's project implementing depth estimation for spherical images using unsupervised learning with CNNs. Based on the work of [Godard et. al](https://github.com/mrharicot/monodepth).

<p align="center">
  <img src="/visualization/synthetic_examples.gif">
</p>

# Requirements

Requires typical scientific computing libraries like numpy, scipy and TensorFlow. Some Python scripts require other libraries like Blender or OpenEXR. The main program for training and testing just needs TensorFlow. Requires a GPU for training in a reasonable amount of time (inference is okay on CPU).

# Structure

- [calibration](calibration)
  - MATLAB scripts for rectifying a top and bottom spherical image.
- [evaluation](evaluation)
  - Code for evaluation predictions against ground truth data. Requires PILLOW, matplotlib, numpy, scipy and OpenEXR.
- [monodepth](monodepth)
  - Main code section which includes a [spherical image module](monodepth/spherical.py) with TensorFlow functions for converting between different projection models. Also includes network definitions and training and testing code. Further instructions are given in the module [README](monodepth/README.md)
- [synthetic](synthetic)
  - Blender scripts for rendering equirectangular images along with ground truth depth (and optionally 3D 360 degree images). Also includes [SUNCGToolbox](https://github.com/srijanparmeshwar/SUNCGtoolbox) for handling data from their paper. Allows you to create a training dataset for the network. Real world data is currently not made publicly available.
- [visualization](visualization)
  - Code for handling outputs from the network i.e. converting from depth map to point cloud, and resampling the depth map.
