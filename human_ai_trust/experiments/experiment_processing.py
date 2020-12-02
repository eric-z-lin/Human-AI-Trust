import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import csv

experiment_id = "9-EZ-L"

header = ["patient_num", "patient_filename","accuracy", "calibration",
				"model_prediction", "ground_truth", "user_prediction", "user_update",
				"question_relationship", "full_questions", "question_perceived_accuracy","question_calibration",
				"question_personal_conf","question_model_conf"]#"question_time"

input_file = csv.DictReader(open('experiment-'+str(experiment_id)+'.csv'))

experiment = {}

for row in input_file:
	patient_num = int(row["patient_num"])
	print(patient_num)
	if(len(row) == len(header)):
		experiment[patient_num] = {}
		for h in header:
			if(h == "accuracy" or h == "calibration"):
				d = json.loads(row[h])
				experiment[patient_num][h] = json.loads(d)
				#print(experiment[patient_num][h]["0"])
			elif(h == "patient_filename"):
				experiment[patient_num][h] = row[h]
			elif(h == "user_update"):
				if(row[h] == "1"):
					experiment[patient_num][h] = 1
				else:
					experiment[patient_num][h] = 0
			else:
				experiment[patient_num][h] = int(row[h])
	else:
		prin("problem")

trends = {}
trends["positive_reinforcement"] = [1 if(experiment[i]["user_update"] == 1 and experiment[i]["user_prediction"] == experiment[i]["model_prediction"]) else 0 for i in range(1,len(experiment.keys())+1)]
trends["negative_reinforcement"] = [1 if(experiment[i]["user_update"] == 1 and experiment[i]["user_prediction"] != experiment[i]["model_prediction"]) else 0 for i in range(1,len(experiment.keys())+1)]
trends["all_questions"] = {}
trends["gt"] = {}
trends["ml_pred"] = {}
trends["user_pred"] = {}

trends["moving_user_acc"] = {}
trends["moving_model_acc"] = {}
trends["moving_model_test_acc_0"] = {}
trends["moving_model_test_acc_1"] = {}


for i in range(1,len(experiment.keys())+1):
	if(experiment[i]["full_questions"] == 1):
		trends["all_questions"][i] = {}
		for j in ["question_perceived_accuracy","question_calibration",
				"question_personal_conf","question_model_conf"]:
			trends["all_questions"][i][j] = experiment[i][j]
		trends["gt"][i] = experiment[i]["ground_truth"]
		trends["ml_pred"][i] = experiment[i]["model_prediction"]
		trends["user_pred"][i] = experiment[i]["user_prediction"]

	if(i >= 5):
		trends["moving_model_acc"][i] = sum([1 if (experiment[j]["model_prediction"] == experiment[j]["ground_truth"]) else 0 for j in range(i-4,i+1)])/5
		trends["moving_user_acc"][i] = sum([1 if (experiment[j]["user_prediction"] == experiment[j]["ground_truth"]) else 0 for j in range(i-4,i+1)])/5
		trends["moving_model_test_acc_0"][i] = experiment[i]["accuracy"]["0"]
		trends["moving_model_test_acc_1"][i] = experiment[i]["accuracy"]["1"]

"""
THINGS TO MEASURE:
	(1) Team performance
	(2) Agreement Percentage
	(3) Positive Reinforcement versus Negative
"""
fig, ax = plt.subplots()

plt.title('User and Model Accuracy over Time')
plt.xlabel('round number')
plt.ylabel('accuracy')
a = matplotlib.patches.Patch(color='black', label='User accurcy')
b = matplotlib.patches.Patch(color='purple', label='Model accuracy')
c = matplotlib.patches.Patch(color='orange', label='Model performance on test for normal')
d = matplotlib.patches.Patch(color='yellow', label='Model performance on test for overexposed')
plt.legend(handles=[a,b,c,d])

#user
arr = [[i, trends["moving_user_acc"][i]] for i in range(5, 5+len(trends["moving_user_acc"].keys()))]
data_user = np.array(arr)
ax.plot([i[0] for i in arr],[i[1] for i in arr], 'o-', color="black", picker=True)
colors = ['black' if (experiment[i]["user_update"] == 0) else 'blue' for i in range(5,5+len(trends["moving_user_acc"].keys()))]
for xy, color in zip(data_user, colors):
	ax.plot(xy[0],xy[1], 'o', color=color, picker=True)

#model
arr = [[i, trends["moving_model_acc"][i]] for i in range(5, 5+len(trends["moving_model_acc"].keys()))]
data_AI = np.array(arr)
ax.plot([i[0] for i in arr],[i[1] for i in arr], 'o-', color="purple", picker=True)
colors = ['black' if (experiment[i]["user_update"] == 0) else 'blue' for i in range(5,5+len(trends["moving_model_acc"].keys()))]
for xy, color in zip(data_AI, colors):
	ax.plot(xy[0],xy[1], 'o', color=color, picker=True)

#model test
arr = [[i, trends["moving_model_test_acc_0"][i]] for i in range(5, 5+len(trends["moving_model_test_acc_0"].keys()))]
data_AI_test = np.array(arr)

ax.plot([i[0] for i in arr],[i[1] for i in arr], 'o-', color="orange", picker=True)
colors = ['black' if (experiment[i]["user_update"] == 0) else 'blue' for i in range(5,5+len(trends["moving_model_test_acc_0"].keys()))]
for xy, color in zip(data_AI_test, colors):
	ax.plot(xy[0],xy[1], 'o', color=color, picker=True)

#model test
arr = [[i, trends["moving_model_test_acc_1"][i]] for i in range(5, 5+len(trends["moving_model_test_acc_1"].keys()))]
data_AI_test = np.array(arr)

ax.plot([i[0] for i in arr],[i[1] for i in arr], 'o-', color="yellow", picker=True)
colors = ['black' if (experiment[i]["user_update"] == 0) else 'blue' for i in range(5,5+len(trends["moving_model_test_acc_1"].keys()))]
for xy, color in zip(data_AI_test, colors):
	ax.plot(xy[0],xy[1], 'o', color=color, picker=True)


plt.show()