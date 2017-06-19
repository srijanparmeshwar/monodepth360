import os, sys

project_path = sys.argv[1]
project_name = sys.argv[2]
path = sys.argv[3]

if project_name == 'suncg':
	obj_path = project_path + "/obj"
	ids = os.listdir(obj_path)
	for id in ids:
		os.system("blender-exp -b " + project_path + "/" + project_name + ".blend -P render.py -- " + path + " " + id + " suncg " + obj_path)
else:
	os.system("blender-exp -b " + project_path  + "/" + project_name + ".blend -P render.py -- " + path + " " + project_name + " blender")