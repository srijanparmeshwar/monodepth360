import bpy, os, sys, time
import numpy as np
from mathutils import *

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from materials_cycles_converter import *

# Utility function to separate list of floats into tuples.
def extract_parameters(parameters):
	location = (parameters[0], - parameters[2], parameters[1])
	direction = (parameters[3], - parameters[5], parameters[4])
	up = (parameters[6], - parameters[8], parameters[7])
	return [location, direction, up]

# Read file of camera extrinsics and intrinsics.
def read_camera_file(path):
	with open(path) as camera_file:
		return [extract_parameters([float(x) for x in line.split()]) for line in camera_file]

# Adapted from https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
def look_at_euler(camera, direction):
	world_camera = camera.matrix_world.to_translation()
	rot_quat = Vector(direction).to_track_quat("-Z", "Y")
	return rot_quat.to_euler()
	
# Add a camera to scene if one does not already exist.
def add_camera():
	scene = bpy.context.scene
	if scene.camera is None:
		bpy.ops.object.camera_add(view_align = False, enter_editmode = False)
		scene.camera = bpy.data.objects["Camera"]

# Set camera location and view direction.
def set_camera(location, direction):
	scene = bpy.context.scene
	camera = scene.camera
	camera.location = location
	camera.rotation_euler = look_at_euler(camera, direction)

# Add Z buffer output node and link to render layer.
def setup_z_buffer():
	scene = bpy.context.scene
	scene.use_nodes = True
	
	nodes = scene.node_tree.nodes

	render_layers = nodes.new("CompositorNodeRLayers")
	z_buffer = nodes.new("CompositorNodeOutputFile")
	#z_buffer.use_alpha = False
	z_buffer.format.file_format = "OPEN_EXR"
	
	scene.node_tree.links.new(
		render_layers.outputs["Depth"],
		z_buffer.inputs[0]
	)

	return z_buffer
	
# Save Z buffer as Numpy file.
def save_z_buffer(width, height, filename):
	blender_z_data = bpy.data.images["Viewer Node"].pixels
	z_data = np.array(blender_z_data)
	print(z_data.shape)
	np.save(filename, z_data)

# Render scene.
def render(name = "output", path = "//render", up = (0, 1, 0), z_buffer = None, width = 1024, height = 512, tile_size = 512, samples = 500):
	scene = bpy.context.scene
	
	# Set output resolution and tile sizes.
	scene.render.resolution_x = width
	scene.render.resolution_y = height
	scene.render.tile_x = tile_size
	scene.render.tile_y = tile_size
	scene.render.image_settings.file_format = "JPEG"
	# scene.render.use_render_cache = True
	scene.render.layers.active.cycles.use_denoising = True
	
	# Settings for Cycles renderer to be faster.
	bpy.context.user_preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
	for device in bpy.context.user_preferences.addons["cycles"].preferences.devices:
		device.use = False
		
	if len(sys.argv) > 10:
		device_index = int(sys.argv[10])
	else:
		device_index = 0
	
	bpy.context.user_preferences.addons["cycles"].preferences.devices[device_index].use = True
	scene.cycles.device = "GPU"
	scene.cycles.samples = samples
	scene.cycles.caustics_reflective = False
	scene.cycles.caustics_refractive = False
	scene.cycles.min_bounces = 2
	scene.cycles.max_bounces = 4
	
	# Set up 360 degree lens.
	camera = scene.camera.data
	camera.type = "PANO"
	camera.cycles.panorama_type = "EQUIRECTANGULAR"

	# Turn off stereo for 2D capture.
	scene.render.use_multiview = False
	z_buffer.base_path = ""
	z_buffer.file_slots[0].path = os.path.join(path, "depth_top", name + ".exr")
	
	scene.render.filepath = os.path.join(path, "top", name)
	bpy.ops.render.render(write_still = True)
	os.rename(os.path.join(path, "depth_top", name + ".exr0001.exr"), os.path.join(path, "depth_top", name + ".exr"))
	
	#save_z_buffer(width, height, os.path.join(path, "depth", name))
	
	scene.camera.location = scene.camera.location - 0.25 * Vector(up)
	
	z_buffer.file_slots[0].path = os.path.join(path, "depth_bottom", name + ".exr")

	scene.render.filepath = os.path.join(path, "bottom", name)
	bpy.ops.render.render(write_still = True)
	os.rename(os.path.join(path, "depth_bottom", name + ".exr0001.exr"), os.path.join(path, "depth_bottom", name + ".exr"))
	
	scene.camera.location = scene.camera.location + 0.25 * Vector(up)
	
	# Turn on stereo for 3D capture.
	# scene.render.use_multiview = True
	# camera.stereo.use_spherical_stereo = True
	
	# scene.render.filepath = path + "/3d/" + name
	# bpy.ops.render.render(write_still = True)

# Get render output arguments.
render_path = sys.argv[6]
name = sys.argv[7]

# Blender project or SUNCG.
if len(sys.argv) < 9:
	mode = "blender"
else:
	mode = sys.argv[8]

if mode == "blender":
	# Just render the scene if it is a Blender project.
	render(name = name, path = render_path)
elif mode == "suncg":
	# Convert materials to Cycles nodes, and add light, then render
	# for each camera.
	suncg_path = sys.argv[9]
	
	# Load scene.
	bpy.ops.import_scene.obj(filepath = suncg_path + "/"  + name + "/house.obj")
	AutoNode()
	
	# Add camera to scene.
	add_camera()
	
	# Load camera parameters.
	print("Reading camera file: " + suncg_path + "/../cameras/"  + name + "/room_camera.txt")
	if os.path.exists(suncg_path + "/../cameras/"  + name + "/room_camera.txt"):
		cameras = read_camera_file(suncg_path + "/../cameras/"  + name + "/room_camera.txt")
	else:
		cameras = []
	
	# Setup Z buffer.
	z_buffer = setup_z_buffer()
	
	# Render for each camera.
	index = 0
	for parameters in cameras:
		location = parameters[0]
		direction = parameters[1]
		up = parameters[2]
		set_camera(location, direction)
		rendered = False
		while not rendered:
			try:
				render(name = name + "_{}".format(index), path = render_path, up = up, z_buffer = z_buffer)
				rendered = True
			except BaseException as error:
				print(error)
				print("Failed to render. Will try again in 120 seconds.")
				time.sleep(120)
		index = index + 1
