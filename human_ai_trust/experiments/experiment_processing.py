import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

experiment_id = 3

header = ["patient_num", "patient_filename","accuracy", "calibration",
				"model_prediction", "ground_truth", "user_prediction", "user_update",
				"question_relationship", "full_questions", "question_perceived_accuracy","question_calibration",
				"question_personal_conf","question_model_conf"]

input_file = csv.DictReader(open('experiments/experiment-'+str(experiment_id)+'.csv'))

experiment = {}

for row in input_file:
    print(row)
    assert(len(row) == len(header))
    experiment[int(row["patient_num"])] = {}
    for h in header:
    	if(h == "accuracy" or h == "calibration"):
    		experiment[row["patient_num"]][h] = json.loads(row[h])
    	elif(h == "patient_filename"):
    		experiment[row["patient_num"]][h] = row[h]
    	else:
    		experiment[row["patient_num"]][h] = int(row[h])

trends = {}
trends["positive_reinforcement"] = [1 if(experiment[i]["user_update"] == 1 and experiment[i]["user_prediction"] == experiment[i]["model_prediction"]) else 0 for i in range(1,len(experiment.keys())+1)]
trends["negative_reinforcement"] = [1 if(experiment[i]["user_update"] == 1 and experiment[i]["user_prediction"] != experiment[i]["model_prediction"]) else 0 for i in range(1,len(experiment.keys())+1)]
trends["all_questions"] = {}

for i in range(1,len(experiment.keys())+1):
	if(experiment[i]["full_questions"] == 1):
		trends["all_questions"][i] = {}
		for j in ["question_perceived_accuracy","question_calibration",
				"question_personal_conf","question_model_conf"]:
			trends["all_questions"][i][j] = experiment[i][j]

"""
THINGS TO MEASURE:
	(1) Team performance
	(2) Agreement Percentage
	(3) Positive Reinforcement versus Negative
"""
fig, ax = plt.subplots()
data = np.array([[4.29488806,-5.34487081],
                [3.63116248,-2.48616998],
                [-0.56023222,-5.89586997],
                [-0.51538502,-2.62569576],
                [-4.08561754,-4.2870525 ],
                [-0.80869722,10.12529582]])
colors = ['red','red','red','blue','red','blue']
for xy, color in zip(data, colors):
    ax.plot(xy[0],xy[1],'o',color=color, picker=True)

plt.show()