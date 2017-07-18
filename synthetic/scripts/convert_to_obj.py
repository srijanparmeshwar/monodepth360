import os

os.chdir("../models/suncg/house/")
ids = os.listdir("./")

for id in ids[0:1000]:
	os.chdir(id)
	print "../../../../SUNCGtoolbox/gaps/bin/x86_64/scn2scn house.json ../../obj/" + id + "/house.obj"
	os.system("../../../../SUNCGtoolbox/gaps/bin/x86_64/scn2scn house.json ../../obj/" + id + "/house.obj")
	print "../../../../SUNCGtoolbox/gaps/bin/x86_64/scn2cam house.json ../../obj/" + id + "/outputcamerasfile -categories ../../../../SUNCGtoolbox/metadata/ModelCategoryMapping.csv -v -debug -glut"
	os.system("../../../../SUNCGtoolbox/gaps/bin/x86_64/scn2cam house.json ../../obj/" + id + "/outputcamerasfile -categories ../../../../SUNCGtoolbox/metadata/ModelCategoryMapping.csv -v -debug -glut")
	os.chdir("../")
