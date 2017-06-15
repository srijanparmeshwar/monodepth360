import bpy, sys
from mathutils import *

# Adapted from https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
def look_at_euler(camera, direction):
    world_camera = camera.matrix_world.to_translation()
    rot_quat = Vector(direction).to_track_quat('-Z', 'Y')
    return rot_quat.to_euler()
	
# Add a camera and set its location and view direction to the provided inputs.
def add_camera(location, direction):
    bpy.ops.object.camera_add(view_align = False, enter_editmode = False, location = location, rotation = (0.0, 0.0, 0.0))
    camera = scene.camera.data
    camera.rotation_euler = look_at_euler(camera, direction)

def render(name = 'output', path = '//render', width = 640, height = 360, tile_size = 512, samples = 144):
    scene = bpy.context.scene

    # Set output resolution and tile sizes.
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.tile_x = tile_size
    scene.render.tile_y = tile_size
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.use_render_cache = True

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
    
output_name = sys.argv[6]
output_path = sys.argv[7]
render(name = output_name, path = output_path)