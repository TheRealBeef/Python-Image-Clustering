# Python Image Clustering (Unsupervised Classifier)

## Command Line Arguments
	-i PATHNAME	# Input directory
	-o FILENAME	# Output .PKL file
	-p NUMBER	# Number of processes for pre-processing
	-s True/False	# Whether to save output and graphs from HOG

## Example Command Line Statements
	
	python3 main.py 		# Runs the classifier with default settings
	python3 main.py -p 16		# Runs the classifier with the specified number of 					     processes. 1-63 are valid inputs
	python3 main.py -s True		# Saves the outputs of HOG and plots their feature 					     graphs

## Non-Standard Libraries Used:
    matplotlib      3.5.2
    opencv-python	4.5.3.56
    scikit-image	0.19.3
    scikit-learn	1.1.1
    scipy	       	1.8.1
    threadpoolctl	3.1.0
    tqdm	        4.64.0
    imutils 	0.5.4

## Outputs
	/temp/*.pkl - These files are temporary and can be ignored. They are removed at the 			  beginning of each run
	/plots/* - The feature plots of images (in no particular order)
	/processed/* - The visualizations of HOG data (in no particular order)
	output.pkl - The consolidated data matrix after pre-processing
	2d_representation_plot.png - 2d State-Space graph of data
	Elbow_graph.png - Graph of # clusters vs variance
