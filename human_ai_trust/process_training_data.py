import os
from glob import glob
import cv2 as cv
import numpy as np
import csv
from PIL import Image

source = "train" #train,val
#source = "val"
#directory = "train_dir_expose_noise_blur" #train_dir_expose_noise_blur, test_dir_expose_noise_blur
directory = "train_dir"
#directory = "test_dir"

def spotlight(img: Image, center: (int, int), radius: int) -> Image:
    width, height = img.size
    overlay_color = (0, 0, 0, 128)
    img_overlay = Image.new(size=img.size, color=overlay_color, mode='RGBA')
    for x in range(width):
        for y in range(height):
            dx = x - center[0]
            dy = y - center[1]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance < radius:
                img_overlay.putpixel((x, y), (0, 0, 0, 0))
    img.paste(img_overlay, None, mask=img_overlay)
    return img

result = [y for x in os.walk("CheXpert-v1.0-small/"+source+"/") for y in glob(os.path.join(x[0], '*.jpg'))]

file_to_diagnosis = {}

if(source == "train"):
	csv_patients = "for_experiment_edema_only.csv"
else:
	csv_patients = 'stat_val_edema_only.csv'

with open (csv_patients) as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	for row in readCSV:
		if("Path" not in row[0]):
			filename = row[0]
			if(source == "train"):
				file = filename
			else:
				file = filename.replace("train","val")
			file_to_diagnosis[file] = int(float(row[6]))

print(len(result))
for counter in range(len(result)):
	filename = result[counter]
	if(source == "train"):
		file = filename
	else:
		file = filename.replace("train","val")
	#old format: CheXpert-v1.0-small/train/patient32584/study3/view1_frontal.jpg
	#new format: img_dir/case-diagnosis-name.png
	file_arr = file.split("/")
	name = str(file_to_diagnosis[file])+"-"+file_arr[-3]+"_"+file_arr[-2]+"_"+file_arr[-1]

	#CASE 0
	case0_image = cv.imread(file)
	
	#CASE 1
	image = cv.imread(file, cv.IMREAD_GRAYSCALE)
		#overexpose
	image2 = cv.convertScaleAbs(image, alpha=1.0, beta=10) 
		#add noise
	uniform_noise = np.zeros((image.shape[0], image.shape[1]),dtype=np.uint8)
	cv.randu(uniform_noise,0,255)
	uniform_noise = (uniform_noise*0.3).astype(np.uint8)
	case1_image = cv.add(image2,uniform_noise)
		#blur
	case1_image = cv.blur(case1_image,(3,3))
	#case1_image = image3


	if(source == "train"):
		cv.imwrite("mainpage/dataset/"+directory+"/"+"0-"+name, case0_image)
	else:
		cv.imwrite("mainpage/dataset/"+directory+"/"+"0-"+name, case0_image)
		cv.imwrite("mainpage/dataset/"+directory+"/"+"1-"+name, case1_image)



