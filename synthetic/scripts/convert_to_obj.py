import os

os.chdir("../models/suncg/house/")
ids = os.listdir("./")

for id_index in range(0, 100):
	id = ids[id_index]
	os.chdir(id)
	#print "../../../../SUNCGtoolbox/gaps/bin/x86_64/scn2scn house.json ../../obj/" + id + "/house.obj"
	#os.system("../../../../SUNCGtoolbox/gaps/bin/x86_64/scn2scn house.json ../../obj/" + id + "/house.obj")
	print "../../../../SUNCGtoolbox/gaps/bin/x86_64/scn2cam house.json ../../obj/" + id + "/outputcamerasfile -categories ../../../../SUNCGtoolbox/metadata/ModelCategoryMapping.csv -v -debug"
	os.system("../../../../SUNCGtoolbox/gaps/bin/x86_64/scn2cam house.json ../../obj/" + id + "/outputcamerasfile -categories ../../../../SUNCGtoolbox/metadata/ModelCategoryMapping.csv -v -debug")
	os.chdir("../")
