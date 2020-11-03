from django.shortcuts import render

from mainpage.models import *

# Create your views here.

from django.http import HttpResponse




def index(request):
	"""View function for home page of site."""

	if request.method == 'GET':
		# Load session Id
		experiment_id = request.session['experiment_id']
		experiment = ModelExperiment.objects.get(id=experiment_id)

		ml_model = experiment.field_model_ml_model

		# Get patient case
		generated_patient = experiment.generate_patient()
		case = experiment.generated_patient_to_case(generated_patient)
		
		arr = ml_model.model_prediction(case)
		model_prediction = arr[0]
		model_confidence = arr[1]
		ground_truth = arr[2]


		initUserResponse = {
			"field_data_point_string": case,
			"field_ml_accuracy":  json.dumps(ml_model.accuracy_field),
			"field_ml_calibration": json.dumps(ml_model.calibration_field),
			"field_ml_prediction": model_prediction,
			"field_ml_confidence": model_confidence,
			"field_instance_ground_truth": ground_truth,
			"field_user_prediction": None,
			"field_user_did_update": None,
			"field_user_disagree_reason_choices": None,
			"field_user_disagree_reason_freetext": None,
		}



		new_user_response = ModelUserResponse(
								field_data_point_string= initUserResponse["field_data_point_string"],
								field_ml_accuracy = initUserResponse["field_ml_accuracy"],
								field_ml_calibration = initUserResponse["field_ml_calibration"],
								field_ml_prediction = initUserResponse["field_ml_prediction"],
								field_instance_ground_truth = initUserResponse["field_instance_ground_truth"],
								field_user_prediction = initUserResponse["field_user_prediction"],
								field_user_did_update = initUserResponse["field_user_did_update"],
								field_user_disagree_reason_choices = initUserResponse["field_user_disagree_reason_choices"],
								field_user_disagree_reason_freetext = initUserResponse["field_user_disagree_reason_freetext"],
								field_experiment = experiment
							)

		new_user_response.save()


		context = {
		    # 'feature_dict': feature_dict,
		    'ml_confidence': model_confidence
		}

		# Render the HTML template index.html with the data in the context variable
		return render(request, 'index.html', context=context)

	if request.method == 'POST':
		print('loaded post')


		context = {}

		return render(request, 'index.html', context=context)



def start_experiment(request):
	"""View function for home page of site."""

	# Instantiate models
	new_experiment = ModelExperiment()
	new_experiment.field_ml_model_accuracy = 1
	new_experiment.field_ml_model_calibration = 0
	new_experiment.field_ml_model_update_type = 0
	ml_model = ModelMLModel()
	ml_model.initialize(
		performance=new_experiment.field_ml_model_accuracy, 
		calibration=new_experiment.field_ml_model_calibration, 
		update=new_experiment.field_ml_model_update_type
	)
	new_experiment.field_model_ml_model = ml_model
	ml_model.save()
	new_experiment.save()

	# Session
	request.session['experiment_id'] = new_experiment.id
	print('session start experiment', request.session['experiment_id'])
	

	# # Generate counts of some of the main objects
	# all_responses = ModelUserResponse.objects.all()
	# print(len(all_responses))
	# num_responses = len(all_responses)

	context = {}

	# Render the HTML template index.html with the data in the context variable
	return render(request, 'start_experiment.html', context=context)


def updateButton(request):
	""" View function for asking questions during update """

	# Check if user agreed with AI
	print()


# def index(request):
#     return HttpResponse("Hello, world. You're at the polls index.")