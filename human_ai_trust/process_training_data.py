import os
from glob import glob
import cv2 as cv
import numpy as np
import csv

result = [y for x in os.walk("CheXpert-v1.0-small/") for y in glob(os.path.join(x[0], '*.jpg'))]

file_to_diagnosis = {}

with open('for_experiment_edema_only.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
    	if("Path" not in row[0]):
    		file_to_diagnosis[row[0]] = int(float(row[6]))

for counter in range(len(result)):
	file = result[counter]
	#old format: CheXpert-v1.0-small/train/patient32584/study3/view1_frontal.jpg
	#new format: img_dir/case-diagnosis-name.png

	file_arr = file.split("/")
	name = str(file_to_diagnosis[file])+"-"+file_arr[-3]+"_"+file_arr[-2]+"_"+file_arr[-1]

	#if(counter < 0.7*len(result)):
	image = cv.imread(file)
	cv.imwrite("mainpage/dataset/train_dir/"+"0-"+name, image)
	#else:
	image = cv.imread(file)
	new_image = cv.convertScaleAbs(image, alpha=1.0, beta=80)
	cv.imwrite("mainpage/dataset/train_dir/"+"1-"+name, new_image)