from django.shortcuts import render

from mainpage.models import *

# Create your views here.

from django.http import HttpResponse




def index(request):
	"""View function for home page of site."""

	# Load session Id
	print('session index', request.session['user_id'])
	


	# Get patient case
	generated_patient = generate_patient()
	case = generated_patient_to_case(generated_patient)
	arr = ml_model.model_prediction(case)
	model_prediction = arr[0]
	model_confidence = arr[1]
	ground_truth = arr[2]


	initUserResponse = {
		"field_data_point_string": '0-0-0-low-low',
		"field_ml_accuracy":  0.5,
		"field_ml_confidence": 0.5,
		"field_ml_prediction": 0,
		"field_user_prediction": 1,
		"field_user_did_update": 1,
		"field_user_disagree_reason_choices": 'a',
		"field_user_disagree_reason_freetext": 'Example long text',
	}



	new_user_response = ModelUserResponse(
							field_data_point_string= initUserResponse["field_data_point_string"],
							field_ml_accuracy = initUserResponse["field_ml_accuracy"],
							field_ml_confidence = initUserResponse["field_ml_confidence"],
							field_ml_prediction = initUserResponse["field_ml_prediction"],
							field_user_prediction = initUserResponse["field_user_prediction"],
							field_user_did_update = initUserResponse["field_user_did_update"],
							field_user_disagree_reason_choices = initUserResponse["field_user_disagree_reason_choices"],
							field_user_disagree_reason_freetext = initUserResponse["field_user_disagree_reason_freetext"]
						)

	new_user_response.save()

	# Generate counts of some of the main objects
	# all_responses = ModelUserResponse.objects.all()
	# print(len(all_responses))
	# num_responses = len(all_responses)


	context = {
	    'user_responses': all_responses,
	    'num_responses': num_responses,
	    'feature_dict': feature_dict,
	}

	# Render the HTML template index.html with the data in the context variable
	return render(request, 'index.html', context=context)



def start_experiment(request):
	"""View function for home page of site."""

	# Instantiate models
	new_experiment = ModelExperiment()
	new_experiment.field_ml_model_accuracy = 0
	new_experiment.field_ml_model_calibration = 0
	new_experiment.field_ml_model_update_type = 0
	ml_model = ModelMLModel()
	ml_model.initialize(
		performance=new_experiment.field_ml_model_accuracy, 
		calibration=new_experiment.field_ml_model_calibration, 
		update=new_experiment.field_ml_model_update_type
	)
	new_experiment.field_model_ml_model_id = ml_model
	ml_model.save()
	new_experiment.save()

	# Session
	request.session['user_id'] = new_experiment.id
	print('session start experiment', request.session['user_id'])
	

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