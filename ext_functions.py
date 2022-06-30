# IMPORTS #
import cv2
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import sklearn.decomposition
from sklearn.cluster import KMeans
from imutils import paths
import numpy as np
import pickle
import os
import skimage.feature
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import defines as d
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

def process_images(payload):

	pool = ThreadPool(1)

	print("[Process %s] started and working on on %s files" % (format(payload["id"]), len(payload["input_paths"])))
	processed = []
	for imagePath in payload["input_paths"]:
		img_name = int(imagePath.split('/')[1][4:8])
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = 255 * (gray < 128).astype(np.uint8)
		coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
		x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
		rect = image[y:y + h, x:x + w]  # Crop the image - note we do this on the original image
		resized = cv2.resize(rect, (d.img_size, d.img_size))

		data = ()
		data = skimage.feature.hog(resized, orientations=9, pixels_per_cell=(8, 8),	cells_per_block=(3, 3), feature_vector=True,
								   visualize=payload["save_images"], channel_axis=2)

		final=(img_name, data)
		processed.append(final)

	print("[Process %s] writing data to %s" % (format(payload["id"]), format(payload["output_path"])))
	f = open(payload["output_path"], "wb")
	f.write(pickle.dumps(processed))
	f.close()
	print("[Process %s] complete" % format(payload["id"]))


def chunk(l, n):
	# loop over the list in n-sized chunks
	for i in range(0, len(l), n):
		# yield the current n-sized chunk to the calling function
		yield l[i: i + n]


def pre_processing(input_directory, working_directory, output_file, num_processes, save):
	allImagePaths = sorted(list(paths.list_images(input_directory)))
	numImagesPerProc = len(allImagePaths) / float(num_processes)
	numImagesPerProc = int(np.ceil(numImagesPerProc))
	chunkedPaths = list(chunk(allImagePaths, numImagesPerProc))

	payloads = []
	for (i, imagePaths) in enumerate(chunkedPaths):
		# construct the path to the output intermediary file for the current process
		outputPath = os.path.sep.join([working_directory, "proc_{}.pkl".format(i)])

		# construct a dictionary of data for the payload, then add it to the payloads list
		data = {
			"id": i,
			"input_paths": imagePaths,
			"output_path": outputPath,
			"save_images": save
		}
		payloads.append(data)

	# construct and launch the processing pool
	print("Preprocessing is using %s processes" % num_processes)
	pool = Pool(processes=num_processes)

	pool.map(process_images, payloads)

	# close the pool and wait for all processes to finish
	pool.close()
	pool.join()
	print("All processes complete")

	print("Combining Outputs")
	images = []
	i = 0

	# loop over all pickle files in the output directory
	for p in paths.list_files(working_directory, validExts=".pkl"):
		print("Loading data from %s" % p)
		stuff = pickle.loads(open(p, "rb").read())  # load the contents of the dictionary
		names, images_output = zip(*stuff)
		if save: print("Writing processed images to /%s/ (%s.png thru %s.png) and histograms to /%s/ (%s.png thru %s.png)"
			  % (d.processed_directory, i, i + len(images_output), d.plots_directory, i, i + len(images_output)))
		j = 0
		for (image) in tqdm(images_output, leave=False):

			if save:
				histogram, plot = image
				plt.imsave(("%s/%s.png" % (d.processed_directory, i)), plot)

				plt.plot(histogram)
				plt.savefig("%s/%s.png" % (d.plots_directory, i))
				plt.clf()
			else:
				histogram = image
			name = names[j]
			results = (name, histogram)
			images.append(results)
			j +=1
			i = i+1

	# serialize the hash dictionary to disk
	print("Writing final output to %s" % output_file)
	f = open(output_file, "wb")
	f.write(pickle.dumps(images))
	f.close()


def print_graph_2d(image_data):
	pca = PCA(n_components=2)
	pca_infos = pca.fit_transform(image_data)
	stuff = KMeans(n_clusters=10, init='k-means++', random_state=0).fit(pca_infos)
	x = []
	y = []
	labels = stuff.labels_
	centers = stuff.cluster_centers_
	variabes = stuff.inertia_
	plt.clf()
	for i in pca_infos:
		x.append(i[0])
		y.append(i[1])
	plt.scatter(x, y, c=labels)
	plt.savefig("test.png")


def print_graph_elbow(image_data):
	pca = PCA(n_components=500)
	pca_infos = pca.fit_transform(image_data)
	elbow_data = []
	for i in tqdm(range(1, 20)):
		elbow_data.append(KMeans(n_clusters=i, init='k-means++', random_state=0).fit(pca_infos))

	x = []
	y = []
	for i in elbow_data:
		x.append(i.n_clusters)
		y.append(i.inertia_)

	plt.plot(x, y)
	plt.savefig("plot2.png")


def compare_labels_kmeans(image_data, names):
	pca = PCA(n_components=500)
	pca_infos = pca.fit_transform(image_data)
	stuff = KMeans(n_clusters=10, init='k-means++', random_state=0).fit(pca_infos)

	group = []



	# assign each image# to a group - 0-99 group 1, 100-199 group 2, etc.
	for i in names:
		group.append(int(i/100))

	print(sklearn.metrics.adjusted_rand_score(group, stuff.labels_))


def cluster_data(data):
	#data un-shitpacking
	names, image_data = zip(*data)

	#print_graph_2d(image_data)
	#print_graph_elbow(image_data)
	compare_labels_kmeans(image_data, names)
