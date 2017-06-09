import os, sys

project_path = sys.argv[1]
project_name = sys.argv[2]
path = sys.argv[3]
os.system("blender -b " + project_path + project_name + ".blend -P render.py -- " + project_name + " " + path)