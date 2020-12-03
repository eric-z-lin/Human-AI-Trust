from django.db import models
import json
import random
import copy
import numpy as np
from PIL import Image
import glob
import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as transforms
# from mainpage.torchhelper import DenseNet121
from torch.utils.data import DataLoader

# from update_model import *
from mainpage.torchhelper import DenseNet121

class ModifiedDataset(Dataset):
	def __init__(self, imgs, transform=None):
		#image file name format: img_dir/case-diagnosis-name.png
		self.image_names = imgs
		self.labels = []
		self.transform = transform

		for file in self.image_names:
			label = [0, 0] # one for each class 
			index = int(file.split("/")[-1].split("-")[1])
			label[index] = 1
			self.labels.append(label)

	def __getitem__(self, index):
		"""Take the index of item and returns the image and its labels"""
		
		image_name = self.image_names[index]
		image = Image.open(image_name).convert('RGB')
		label = self.labels[index]
		if self.transform is not None:
			image = self.transform(image)
		return image, torch.FloatTensor(label)

	def add_data(self, img_filename, label = None, multiplier=1):
		for i in range(multiplier):
			self.image_names.append(img_filename)
			l = [0, 0] # one for each class 
			if(label is None):
				l[int(img_filename.split("/")[-1].split("-")[1])] = 1
			else:
				l[label] = 1
			self.labels.append(l)
		assert(len(self.image_names) == len(self.labels))

	def __len__(self):
		return len(self.image_names)

class ImageDiagnosis:
	def __init__(self, train_dir, test_dir):

		# print(os.getcwd())
		img_dir = './mainpage/dataset/train_dir'
		self.train_imgs = glob.glob(img_dir+'/*.jpg')
		#image file name format: img_dir/case-diagnosis-name.png

		self.feature_names = {'img': "Image"}
		self.features = ['img']
		# self.feature_values = {'img': img_filenames} ?????
		self.feature_values = {'img': self.train_imgs}
		"""
		self.feature_value_names = {}
		for f in self.train_imgs:
			print(f)
			self.feature_value_names[f] = ""
		"""
		test_dir = './mainpage/dataset/test_dir'
		self.test_imgs = glob.glob(test_dir+'/*.jpg')

		self.cases = [0,1]

		imgtransResize = (320, 320)
		imgtransCrop = 224
		normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		transformList = []
		#transformList.append(transforms.Resize(imgtransCrop))
		transformList.append(transforms.RandomResizedCrop(imgtransCrop))
		transformList.append(transforms.RandomHorizontalFlip())
		transformList.append(transforms.ToTensor())
		transformList.append(normalize)      
		transformSequence=transforms.Compose(transformList)

		self.transformSequence = transformSequence

	def ground_truth(self, img_filename):
		img_name = img_filename.split("/")[-1]
		case = int(img_name.split("-")[0])
		gt = int(img_name.split("-")[1])
		return (gt, case)

class ModelMLModel(models.Model):
	domain = ImageDiagnosis("dataset/train_dir", "dataset/test_dir")
	accuracy_field = models.TextField(blank=True, null=True, default='{}')
	model_field = models.BinaryField(default=b'"\'"')
	calibration_field = models.TextField(blank=True, null=True, default='{}')
	update_type_field = models.IntegerField()
	batched_accuracy_field = models.TextField(blank=True, null=True, default='{}')
	batched_model_field = models.BinaryField(default=b'"\'"')

	# Metadata
	class Meta: 
		ordering = ['-accuracy_field', '-calibration_field', '-update_type_field', '-batched_accuracy_field']

	def model_inference(self, img_filename, batched=0):
		if(batched == 1):
			model = pickle.loads(self.batched_model_field)
		else:
			model = pickle.loads(self.model_field)

		dataset = ModifiedDataset([img_filename], self.domain.transformSequence)
		dataLoader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

		prediction = -1

		model.eval()
		with torch.no_grad():
			for i, (input, target) in enumerate(dataLoader):
				bs, c, h, w = input.size()
				varInput = input.view(-1, c, h, w).cuda(non_blocking=True)
				target = target.cuda(non_blocking = True)

				out = model(varInput)
				# pred = (out>0.5).float()
				# prediction = pred.item()
				_, predicted = torch.max(out.data, 1)

				_, label = torch.max(target.data, 1)

				print("output of model", out.data)

				prediction = predicted.data[-1].item()
				raw_output = out.data[-1][prediction].item()

				print("model prediction: ", prediction, ", confidence: ", raw_output)

				return prediction, raw_output

				# if(abs(prediction - 1) < 0.001):
				# 	return prediction, out
				# else:
				# 	return prediction, 1-out

	def model_inference_case(self, case, batched=0):
		
		if(batched == 1):
			model = pickle.loads(self.batched_model_field)
		else:
			model = pickle.loads(self.model_field)
		
		#image file name format: img_dir/case-diagnosis-name.png
		# test = [(img if (('/'+str(case)+'-') in img)) for img in self.domain.test_imgs]
		test = []

		for img in self.domain.test_imgs:
			if ('/'+str(case)+'-') in img:
				test.append(img)

		dataset = ModifiedDataset(test, self.domain.transformSequence)
		dataLoader = DataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

		correct = 0
		example_counter = 0
		with torch.no_grad():
			for i, (input, target) in enumerate(dataLoader):
				target = target.cuda(non_blocking=True)
				input = input.cuda(non_blocking=True)

				bs, c, h, w = input.size()
				varInput = input.view(-1, c, h, w)
			
				out = model(varInput)
				pred = (out>0.5).float()
				correct += (pred == target).float().sum().item()/2

				assert(pred.shape[0] == target.shape[0])
				example_counter += target.shape[0]
		return (correct + 0.0)/ example_counter

	def model_finetune(self, dataset, epochs=1):
		model = pickle.loads(self.batched_model_field)

		for param in model.densenet121.parameters():
		    param.requires_grad = True
		for param in model.densenet121.fc.parameters():
		    param.requires_grad = True

		dataLoaderTrain = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

		model.train()
		optimizer = optim.Adam (model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
		loss = torch.nn.BCELoss(size_average = True)

		for e in range(epochs):
			for batchID, (varInput, target) in enumerate(dataLoaderTrain):
				varTarget = target.cuda(non_blocking = True)
				varInput = varInput.cuda(non_blocking = True)

				varOutput = model(varInput)
				lossvalue = loss(varOutput, varTarget)
				
				optimizer.zero_grad()
				lossvalue.backward()
				optimizer.step()

		self.batched_model_field = pickle.dumps(model)

	def initialize(self, calibration, update, model_pickle_file):
		# gpu_model = torch.load(model_pickle_file, map_location=torch.device('cpu'))
		nClasses = 2
		# model = DenseNet121(nClasses)
		# model = gpu_model.module 

		model = DenseNet121(2).cuda()
		#model = torch.nn.DataParallel(model).cuda()
		print('model initialized')
		gpu_model = torch.load(model_pickle_file)

		from collections import OrderedDict
		new_state_dict = OrderedDict()
		for k, v in gpu_model.items():
		    name = k.replace('module.','') # remove `module.`
		    new_state_dict[name] = v

		model.load_state_dict(new_state_dict)
		device = torch.device("cuda")
		model = model.to(device)
		print('model loaded')

		# model = DenseNet121(nClasses).cuda()
		# model.load_state_dict(torch.load(model_pickle_file))
		# device = torch.device("cuda")
		# model = model.to(device)

		self.model_field = pickle.dumps(model)
		# self.model_field = pickle.dumps(pickle.loads(open(model_pickle_file, "rb")))
		self.batched_model_field = self.model_field

		accuracy = {} #stores the accuracy for case 0 and case 1
		for case in self.domain.cases:
			accuracy[str(case)] = self.model_inference_case(case,batched=0)

		good_calibration_std = 0.03
		bad_calibration_std = 0.05

		calibration_dict = {} #stores the standard deviation
		calibration=int(calibration)
		if(calibration == 1):
			for case in self.domain.cases:
				calibration_dict[case] = good_calibration_std
		else: #calibration == 0, not well-calibrated for all
			for case in self.domain.cases:
				calibration_dict[case] = bad_calibration_std

		print('cal dict')
		print(calibration_dict)

		batched_accuracy = copy.deepcopy(accuracy)
		
		self.update_type_field = update
		self.calibration_field = json.dumps(calibration_dict)

		self.accuracy_field = json.dumps(accuracy)
		self.batched_accuracy_field = json.dumps(batched_accuracy)

	def batch_update(self):
		self.accuracy_field = self.batched_accuracy_field
		self.model_field = self.batched_model_field

	def model_update(self, img_filename, model_prediction, user_prediction, gt):
		batched_accuracy = json.loads(self.batched_accuracy_field)
		calibration = json.loads(self.calibration_field)

		case = int(img_filename.split("/")[-1].split("-")[0])
		transform = self.domain.transformSequence

		if(model_prediction == user_prediction): #user and model agreed, calibration update as well
			case = str(case)
			if(user_prediction != gt): #user and model were incorrect, calibration decreases
				calibration[case] = min(calibration[case]*1.05, 0.1)
			else: #user and model were correct, calibration increases
				calibration[case] = min(calibration[case]*0.9, 0.1)
			mult = len(self.domain.train_imgs)//10
		else: #(model_prediction != user_prediction): #user and model disagreed, accuracy update
			mult = len(self.domain.train_imgs)//7
		dataset = ModifiedDataset(self.domain.train_imgs, transform)
		dataset.add_data(img_filename, user_prediction, multiplier=mult)

		#print("dataset,", dataset.image_names)
		#print("dataset labels,", dataset.labels)

		self.model_finetune(dataset, epochs=3)
		for case in self.domain.cases: #update the accuracy
			batched_accuracy[str(case)] = self.model_inference_case(case, batched=1)
			print(case)
			print(batched_accuracy[str(case)])
		self.batched_accuracy_field = json.dumps(batched_accuracy)
		self.calibration_field = json.dumps(calibration)

		if(self.update_type_field == 1 or self.update_type_field == 3):
			self.batch_update()

	def model_prediction(self, img_filename):
		gt,case = self.domain.ground_truth(img_filename)

		accuracy = json.loads(self.accuracy_field)
		calibration = json.loads(self.calibration_field)

		prediction, model_conf = self.model_inference(img_filename, batched=0)

		confidence = list(np.random.normal(model_conf, calibration[str(case)], 1))[0]
		confidence = min(max(confidence, 0.5000),1.0)
		return [prediction, confidence, gt]

class ModelExperiment(models.Model):
	"""A typical class defining a model, derived from the Model class."""

	# linking fields

	# Fields
	# 0 or 1
	field_ml_model_calibration = models.IntegerField(help_text="0: Poor calibration, 1: Good calibration, 2: No confidence displayed")
	# 0, 1, 2, or 3
	field_ml_model_update_type = models.IntegerField(help_text="0: Control/no update, 1: instant update, 2: batched update, 3: active learning")

	field_user_name = models.CharField(max_length=40, blank=True, help_text='User name')

	field_patient_number = models.IntegerField(help_text="from 0 to 30", default=0)

	# Link an ML model to this experiment
	field_model_ml_model = models.ForeignKey(ModelMLModel, on_delete=models.SET_NULL, blank=True, null=True)

	field_score = models.IntegerField(help_text="Current experiment score", default=0)

	domain = ImageDiagnosis("dataset/train_dir", "dataset/test_dir")

	field_patient_ordering = models.TextField(blank=True, null=True, default='{}')

	def set_ordering(self):
		num_trials = 30
		num_buckets = 3
		# splits = [[1,0],[0.75,0.25],[0.5,0.5],[0.25,0.75],[0,1]]
		splits = [[1,0],[0.5,0.5],[0,1]]
		patients_per_bucket = num_trials / num_buckets

		#image file name format: img_dir/case-diagnosis-name.png
		normal = [i for i in self.domain.test_imgs if (int(i.split("/")[-1].split("-")[0]) == 0)]
		overexposed = [i for i in self.domain.test_imgs if (int(i.split("/")[-1].split("-")[0]) == 1)]
		random.shuffle(normal)
		random.shuffle(overexposed)

		ordering = []

		for i in range(num_buckets):
			norm_patients = round(splits[i][0]*patients_per_bucket)
			over_patients = max(round(patients_per_bucket - norm_patients), 0)

			l = []
			for p in range(norm_patients):
				random_index = np.random.randint(0, len(normal))
				l.append(normal.pop(random_index))
			
			for p in range(over_patients):
				random_index = np.random.randint(0, len(overexposed))
				l.append(overexposed.pop(random_index))

			random.shuffle(l)
			ordering.extend(l)

		final_order = {}
		for i in range(len(ordering)):
			final_order[i] = ordering[i]

		#print(final_order)
		assert(len(ordering) == num_trials)

		self.field_patient_ordering = json.dumps(final_order)

	def generate_patient(self):
		final_order = json.loads(self.field_patient_ordering)
		generated_patient = final_order[str(self.field_patient_number)]
		print('patient',generated_patient)
		return generated_patient

class ModelUserResponse(models.Model):
	"""A typical class defining a model, derived from the Model class."""

	# has an id field from models.Model

	# Binary Features: presence of cough, chills, tested positive for flu
	# Categorical: blood pressure (high/med/low)
	# Continuous Features: body temp (>105, 105-95, 95>), weight (normal = 100)


	# At every time step
	USER_RELATIONSHIP_RESPONSES = (
		(1, '1. I only use my own knowledge.'),
		(2, '2.'),
		(3, '3. I use the AI\'s prediction and my own knowledge equally.'),
		(4, '4.'),
		(5,'5. I only use the AI\'s prediction.'),
	)
	field_user_relationship = models.IntegerField(null=True, choices=USER_RELATIONSHIP_RESPONSES, blank=True, 
				default=3, help_text='Measure of user relationship with model')

	# At every 10% interval time step
	USER_TRUST_RESPONSES = (
		(1, "Strongly disagree"),
		(2, "Disagree"),
		(3, "Neutral"),
		(4, "Agree"),
		(5, "Strongly agree"),
	)
	field_user_perceived_accuracy = models.IntegerField(null=True, choices=USER_TRUST_RESPONSES, blank=True, 
				default=3, help_text='Measure of perceived accuracy')
	field_user_calibration = models.IntegerField(null=True, choices=USER_TRUST_RESPONSES, blank=True, 
				default=3, help_text='Measure of confidence calibration')
	field_user_personal_confidence = models.IntegerField(null=True, choices=USER_TRUST_RESPONSES, blank=True, 
				default=3, help_text='Measure of confidence user in their own mental model')
	field_user_AI_confidence = models.IntegerField(null=True, choices=USER_TRUST_RESPONSES, blank=True, 
				default=3, help_text='Measure of confidence / trust in AI')
	field_user_start_time = models.IntegerField(null=True, help_text='Start epoch time')
	field_user_end_time = models.IntegerField(null=True, help_text='End epoch time')

	# linking fields
	field_data_point_string = models.CharField(max_length=20, help_text="Unique string to specify the input feature combo")

	# Fields
	# field_ml_accuracy = models.DecimalField(max_digits=4, decimal_places=4, help_text="ML accuracy at time of question")
	field_ml_accuracy = models.TextField(blank=True, null=True, default='{}', help_text="ML accuracy at time of question -- json dict")
	field_ml_calibration = models.TextField(blank=True, null=True, default='{}', help_text="ML calibration at time of question -- json dict")
	field_ml_prediction = models.IntegerField(null=True, help_text="Actual ML prediction")
	field_ml_confidence = models.DecimalField(null=True, help_text="Actual ML confidence", decimal_places=5, max_digits=10)
	field_instance_ground_truth = models.IntegerField(null=True, help_text="Ground truth label")

	field_user_prediction = models.IntegerField(null=True, help_text="User prediction")
	field_user_did_update = models.IntegerField(null=True, help_text="Whether or not user updated the model")

	# field_user_disagree_reason_choices = models.CharField(null=True, max_length=1, choices=USER_DISAGREE_REASON_RESPONSES, blank=True, 
	# 			default='a', help_text='If user does not use model prediction, ask why')
	# field_user_disagree_reason_freetext = models.TextField(null=True, help_text='If user chose "other", provide freetext box')

	# field_user_trust = models.IntegerField(null=True, choices=USER_TRUST_RESPONSES, blank=True, 
	# 			default=0, help_text='Measure of how much user trusts model')


	field_experiment = models.ForeignKey(ModelExperiment, on_delete=models.SET_NULL, blank=True, null=True)

	# # Metadata
	# class Meta: 
	# 	ordering = ['-my_field_name']

	# Methods
	def get_absolute_url(self):
		"""Returns the url to access a particular instance of MyModelName."""
		return reverse('model-detail-view', args=[str(self.id)])

	def __str__(self):
		"""String for representing the MyModelName object (in Admin site etc.)."""
		return self.field_experiment


