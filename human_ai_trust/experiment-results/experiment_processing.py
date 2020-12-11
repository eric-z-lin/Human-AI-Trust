import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import csv

from matplotlib.legend_handler import HandlerPatch

class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = matplotlib.patches.Ellipse(xy=center, width=height + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

experiment_group_dict = {
		"experiment-1-Sam.csv": "Poor calibration, No Updates",
		"experiment-2-Carolyn.csv": "Good calibration, No Updates",
		"experiment-3-Junu.csv": "Poor calibration, Batched Updates",
		"experiment-5-Tony.csv": "Poor calibration, Immediate Updates",
		"experiment-6-Henry.csv": "Poor calibration, Batched Updates",
		"experiment-7-Chelsea.csv": "Good calibration, Immediate Updates",
		"experiment-8-Dan.csv": "Poor calibration, Immediate Updates",
		"experiment-9-Angela.csv": "Good calibration, Immediate Updates",
		"experiment-10-Austin.csv": "Good calibration, Batched Updates",
		"experiment-12-Matthew.csv": "Good calibration, No Updates",
		"experiment-13-Neeyanth.csv": "Good calibration, Batched Updates",
		"experiment-14-Catherine.csv": "Poor calibration, No Updates",
		"experiment-15-Joyce.csv": "Good calibration, No Updates",
		"experiment-16-Victor.csv": "Good calibration, Immediate Updates",
	}
header = ["patient_num", "patient_filename","accuracy", "calibration",
					"model_prediction", "ground_truth", "user_prediction", "user_update",
					"question_relationship", "full_questions", "question_perceived_accuracy","question_calibration",
					"question_personal_conf","question_model_conf","time_passed","time_stamp","user_name"]#"question_time"

experiment_ids = list(experiment_group_dict.keys())
for experiment_id in experiment_ids:
	input_file = csv.DictReader(open(experiment_id))

	experiment = {}

	for row in input_file:
		patient_num = int(row["patient_num"])
		#print(patient_num)
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
				elif(h == "time_passed" or h == "time_stamp" or h == "user_name"):
					experiment[patient_num][h] = row[h]
				else:
					experiment[patient_num][h] = int(row[h])
		else:
			print("problem")

	trends = {}
	trends["positive_reinforcement"] = [1 if(experiment[i]["user_update"] == 1 and experiment[i]["user_prediction"] == experiment[i]["model_prediction"]) else 0 for i in range(1,len(experiment.keys())+1)]
	trends["negative_reinforcement"] = [1 if(experiment[i]["user_update"] == 1 and experiment[i]["user_prediction"] != experiment[i]["model_prediction"]) else 0 for i in range(1,len(experiment.keys())+1)]
	trends["all_questions"] = {}
	trends["gt"] = {}
	trends["ml_pred"] = {}
	trends["user_pred"] = {}

	trends["moving_user_acc"] = [0 for i in range(0,31)]
	trends["moving_model_acc"] = [0 for i in range(0,31)]
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

		if(experiment[i]["model_prediction"] == experiment[i]["ground_truth"]):
			trends["moving_model_acc"][i] = 1
		else:
			trends["moving_model_acc"][i] = 0

		if(experiment[i]["user_prediction"] == experiment[i]["ground_truth"]):
			trends["moving_user_acc"][i] = 1
		else:
			trends["moving_user_acc"][i] = 0

		trends["moving_model_test_acc_0"][i] = experiment[i]["accuracy"]["0"]
		trends["moving_model_test_acc_1"][i] = experiment[i]["accuracy"]["1"]

	"""
	THINGS TO MEASURE:
		(1) Team performance
		(2) Agreement Percentage
		(3) Positive Reinforcement versus Negative
	"""
	fig = plt.figure()
	ax = plt.subplot(111)

	version = experiment_group_dict[experiment_id]
	print(version, experiment_id)
	print(sum(trends["moving_user_acc"])*20-40*(30-sum(trends["moving_user_acc"])))
	print(sum([1 if (trends["moving_user_acc"][i] == trends["moving_model_acc"][i]) else 0 for i in range(1, 31)]))
	plt.title("P"+experiment_id.split("-")[1] + ": " + version)
	plt.xlabel('Round Number')
	plt.ylabel('Accuracy')
	plt.ylim(0, 1.2)
	a = matplotlib.patches.Patch(color='black', label='User accuracy')
	b = matplotlib.patches.Patch(color='green', label='Model accuracy')
	c = matplotlib.patches.Patch(color='orange', label='Model performance on test for normal')
	d = matplotlib.patches.Patch(color='purple', label='Model performance on test for degraded')
	e = matplotlib.patches.Circle((0.5, 0.5), radius = 0.25,color='green', label='Model and User disagreed, User correct')
	f = matplotlib.patches.Circle((0.5, 0.5), radius = 0.25,color='red', label='Model and User disagreed, User incorrect')
	g = matplotlib.patches.Circle((0.5, 0.5), radius = 0.25,color='lightblue', label='User updated and correct')
	h = matplotlib.patches.Circle((0.5, 0.5), radius = 0.25,color='darkblue', label='User updated and incorrect')
	

	plt.legend(handles=[a,b,c,d,e,f,g,h], prop={'size': 8}, loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=2, handler_map={matplotlib.patches.Circle: HandlerEllipse()})

	fig.subplots_adjust(bottom=0.3)

	#print(experiment_id.split("-")[1], trends["moving_user_acc"])
	#print(experiment_id.split("-")[1], trends["moving_model_acc"])
	#user
	arr = [[i, np.average(trends["moving_user_acc"][i-4:i+1], weights=[0.1,0.2,0.3,0.4,0.5])] for i in range(5, 31)]
	data_user = np.array(arr)
	agree = [True if (experiment[i]["model_prediction"] == experiment[i]["user_prediction"]) else False for i in range(5,31)]
	ax.plot([i[0] for i in arr],[i[1] for i in arr], '-', color="black", picker=True)
	for index,(xy, color) in enumerate(zip(data_user, agree)):
		if(color == False and trends["moving_user_acc"][index+5] == 1):
			ax.plot(xy[0],xy[1], 'o', color="green", picker=True)
		elif(color == False and trends["moving_user_acc"][index+5] == 0):
			ax.plot(xy[0],xy[1], 'o', color="red", picker=True)

	#model
	arr = [[i, np.average(trends["moving_model_acc"][i-4:i+1], weights=[0.1,0.2,0.3,0.4,0.5])] for i in range(5, 31)]
	data_AI = np.array(arr)
	ax.plot([i[0] for i in arr],[i[1] for i in arr], '-', color="green", picker=True)
	agree = [True if (experiment[i]["model_prediction"] == experiment[i]["user_prediction"]) else False for i in range(5,31)]
	"""
	for index,(xy, color) in enumerate(zip(data_AI, agree)):
		if(color == False and trends["moving_model_acc"][index+5] == 1):
			ax.plot(xy[0],xy[1], 'o', color="green", picker=True)
		elif(color == False and trends["moving_model_acc"][index+5] == 0):
			ax.plot(xy[0],xy[1], 'o', color="red", picker=True)
	"""
	#model test
	arr = [[i-1, trends["moving_model_test_acc_0"][i]] for i in range(1, 31)]
	data_AI_test = np.array(arr)

	ax.plot([i[0] for i in arr],[i[1] for i in arr], '-', color="orange", picker=True)
	colors = [True if (int(experiment[i]["user_update"]) == 1) else False for i in range(1,1+len(trends["moving_model_test_acc_0"].keys()))]
	for index,(xy, color) in enumerate(zip(data_AI_test, colors)):
		#print(index, color, "-",experiment[index+1]["user_update"],"-",int(experiment[i]["user_update"]))
		if(index > 0 and color == True and "No Updates" not in version and "/0-" in experiment[index+1]["patient_filename"]):
			if(trends["moving_model_acc"][index+1] == 1):
				ax.plot(xy[0],xy[1], 'o', color="lightblue", picker=True)
			else:
				ax.plot(xy[0],xy[1], 'o', color="darkblue", picker=True)

	#model test
	arr = [[i-1, trends["moving_model_test_acc_1"][i]] for i in range(1, 31)]
	data_AI_test = np.array(arr)

	ax.plot([i[0] for i in arr],[i[1] for i in arr], '-', color="purple", picker=True)
	colors = [True if (int(experiment[i]["user_update"]) == 1) else False for i in range(1,1+len(trends["moving_model_test_acc_1"].keys()))]
	for index,(xy, color) in enumerate(zip(data_AI_test, colors)):
		#print(index, color, "-",experiment[index+1]["user_update"],"-",int(experiment[i]["user_update"]))
		if(index > 0 and color == True and "No Updates" not in version and "/1-" in experiment[index+1]["patient_filename"]):
			if(trends["moving_model_acc"][index+1] == 1):
				ax.plot(xy[0],xy[1], 'o', color="lightblue", picker=True)
			else:
				ax.plot(xy[0],xy[1], 'o', color="darkblue", picker=True)


	plt.savefig("P"+experiment_id.split("-")[1]+".png")