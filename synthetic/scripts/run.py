import os, sys

project_path = sys.argv[1]
project_name = sys.argv[2]
path = sys.argv[3]
start = int(sys.argv[4])
end = int(sys.argv[5])
device_index = sys.argv[6]

if project_name == 'suncg':
	obj_path = project_path + "/obj"
	ids = os.listdir(obj_path)
	for id in ids[start:end]:
		os.system("../../../blender/blender -b " + project_path + "/" + project_name + ".blend -P render.py -- " + path + " " + id + " suncg " + obj_path + " " + device_index)
else:
	os.system("../../../blender/blender -b " + project_path  + "/" + project_name + ".blend -P render.py -- " + path + " " + project_name + " blender")