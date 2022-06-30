# IMPORTS #
import os
import defines as d
import ext_functions as ext
import argparse
from multiprocessing import cpu_count
import pickle

def remove_directory(directory_name):
	try:
		for filepath in os.listdir(directory_name):
			os.remove(directory_name + "/{}".format(filepath))
		os.rmdir(directory_name)
		print("Folder /%s/  Removed" % directory_name)
	except:
		print("Working folder doesn't exist!!??!!")

if __name__ == "__main__":

	# Step 1: Establish Arguments

	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--images", required=False, type=str, help="name of image directory")
	ap.add_argument("-a", "--output", required=False, type=str, help="name of output pkl")
	ap.add_argument("-p", "--procs", required=False, type=int, help="# of processes to use")
	ap.add_argument("-s", "--save", required=False, type=bool, help="save HOG images to processed directory")
	args = vars(ap.parse_args())

	if not args["images"]: args["images"] = "Cluster_img"
	if not args["output"]: args["output"] = "output.pkl"
	if not args["save"]: args["save"] = False
	if not args["procs"]: args["procs"] = 1 #(cpu_count() * 4)



	remove_directory(d.working_directory)
	try: os.mkdir(d.working_directory)
	except: print("Working directory exists...")
	if args["save"]:
		remove_directory(d.plots_directory)
		try: os.mkdir(d.plots_directory)
		except: print("Plots directory exists...")
		remove_directory(d.processed_directory)
		try: os.mkdir(d.processed_directory)
		except: print("Processed images directory exists...")

	# Step 2: Load Image and Perform Pre-Processing (Color, Scale, resize, etc)
	#ext.pre_processing(args["images"], d.working_directory, args["output"], args["procs"], args["save"])

	# Step 3: Extract Features from Images (PCA, HOG, etc)
	# currently in pre_processing

	# Step 4: Cluster Images
	data = ()
	data = pickle.loads(open(args["output"], "rb").read())
	ext.cluster_data(data)
	# Step 5: Label and graph the clusters

	remove_directory(d.working_directory)
	print("Completed Successfully")