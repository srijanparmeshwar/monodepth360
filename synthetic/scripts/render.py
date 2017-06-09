import bpy, sys

def render(name = 'output', path = '//render', width = 512, height = 288, tile_size = 512, samples = 128):
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