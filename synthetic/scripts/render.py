import bpy, os, sys
from mathutils import *
from materials_cycles_converter import *

# Utility function to separate list of floats into tuples.
def extract_parameters(parameters):
	location = (parameters[0], - parameters[2], parameters[1])
	direction = (parameters[3], - parameters[5], parameters[4])
	return [location, direction]

# Read file of camera extrinsics and intrinsics.
def read_camera_file(path):
	with open(path) as camera_file:
		return [extract_parameters([float(x) for x in line.split()]) for line in camera_file]

# Adapted from https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
def look_at_euler(camera, direction):
	world_camera = camera.matrix_world.to_translation()
	rot_quat = Vector(direction).to_track_quat('-Z', 'Y')
	return rot_quat.to_euler()
	
# Add a camera to scene if one does not already exist.
def add_camera():
	scene = bpy.context.scene
	if scene.camera is None:
		bpy.ops.object.camera_add(view_align = False, enter_editmode = False)
		scene.camera = bpy.data.objects['Camera']

# Set camera location and view direction.
def set_camera(location, direction):
	scene = bpy.context.scene
	camera = scene.camera
	camera.location = location
	camera.rotation_euler = look_at_euler(camera, direction)

# Render scene.
def render(name = 'output', path = '//render', width = 1024, height = 512, tile_size = 1024, samples = 600):
	scene = bpy.context.scene
	
	# Set output resolution and tile sizes.
	scene.render.resolution_x = width
	scene.render.resolution_y = height
	scene.render.tile_x = tile_size
	scene.render.tile_y = tile_size
	scene.render.image_settings.file_format = 'JPEG'
	scene.render.use_render_cache = True
	# scene.render.layers.active.cycles.use_denoising = True
	
	# Settings for Cycles renderer to be faster.
	scene.cycles.device = 'GPU'
	scene.cycles.samples = samples
	scene.cycles.caustics_reflective = False
	scene.cycles.caustics_refractive = False
	scene.cycles.min_bounces = 2
	scene.cycles.max_bounces = 4
	
	# Set up 360 degree lens.
	camera = scene.camera.data
	camera.type = 'PANO'
	camera.cycles.panorama_type = 'EQUIRECTANGULAR'

	# Turn off stereo for 2D capture.
	scene.render.use_multiview = False
	
	scene.render.filepath = path + '/2d/' + name
	bpy.ops.render.render(write_still = True)
	
	# Turn on stereo for 3D capture.
	scene.render.use_multiview = True
	camera.stereo.use_spherical_stereo = True
	
	scene.render.filepath = path + '/3d/' + name
	bpy.ops.render.render(write_still = True)

# Get render output arguments.
render_path = sys.argv[6]
name = sys.argv[7]

# Blender project or SUNCG.
if len(sys.argv) < 9:
	mode = 'blender'
else:
	mode = sys.argv[8]

if mode == 'blender':
	# Just render the scene if it is a Blender project.
	render(name = name, path = render_path)
elif mode == 'suncg':
	# Convert materials to Cycles nodes, and add light, then render
	# for each camera.
	suncg_path = sys.argv[9]
	
	# Load scene.
	bpy.ops.import_scene.obj(filepath = suncg_path + '/'  + name + '/house.obj')
	AutoNode()
	
	# Add camera to scene.
	add_camera()
	
	# Load camera parameters.
	print('Reading camera file: ' + suncg_path + '/../cameras/'  + name + '/room_camera.txt')
	cameras = read_camera_file(suncg_path + '/../cameras/'  + name + '/room_camera.txt')
	
	# Render for each camera.
	index = 0
	for parameters in cameras:
		location = parameters[0]
		direction = parameters[1]
		set_camera(location, direction)
		render(name = name + '_' + str(index), path = render_path)
		index = index + 1