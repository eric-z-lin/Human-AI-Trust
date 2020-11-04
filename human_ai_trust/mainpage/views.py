from django.shortcuts import render, redirect
from django import forms

from mainpage.models import *

# Create your views here.

from django.http import HttpResponse, HttpResponseRedirect




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
		model_confidence = round(arr[1]*100, 1)
		ground_truth = arr[2]

		# Table for patient case
		domain = ml_model.domain
		feature_display_dict = {}
		for feat in domain.features:
			feature_display_dict[domain.feature_names[feat]] = domain.feature_value_names[feat + "-" + generated_patient[feat]]


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
		    'ml_model_prediction': ('Negative' if model_prediction == 0 else 'Positive'),
		    'ml_confidence': str(model_confidence) + "%",
		    'feature_display_dict': feature_display_dict,
		    'user_response': new_user_response,
		}

		# Render the HTML template index.html with the data in the context variable
		return render(request, 'index.html', context=context)

	if request.method == 'POST':
		print('loaded post')


		context = {}

		return render(request, 'index.html', context=context)



class InitExperimentForm(forms.Form):
	user_name = forms.CharField(label='Your name', max_length=100)
	ACCURACY_CHOICES =( 
		(0, "Low"),
		(1, "High")
	)
	field_ml_model_accuracy = forms.ChoiceField(choices = ACCURACY_CHOICES)

	CALIBRATION_CHOICES =( 
		(0, "Low"),
		(1, "High")
	)
	field_ml_model_calibration = forms.ChoiceField(choices = CALIBRATION_CHOICES)

	UPDATE_TYPE_CHOICES =( 
		(0, "None"),
		(1, "Immediate"),
		(2, "Batched")
	)
	field_ml_model_update_type = forms.ChoiceField(choices = UPDATE_TYPE_CHOICES)


def start_experiment(request):
	"""View function for home page of site."""

	# The request method 'POST' indicates
	# that the form was submitted
	if request.method == 'POST':  # 1
		# Create a form instance with the submitted data
		form = InitExperimentForm(request.POST)  # 2
		# Validate the form
		print('checking validity')
		if form.is_valid(): 

			# Instantiate models
			new_experiment = ModelExperiment()
			new_experiment.field_ml_model_accuracy = form.cleaned_data['field_ml_model_accuracy']
			new_experiment.field_ml_model_calibration = form.cleaned_data['field_ml_model_calibration']
			new_experiment.field_ml_model_update_type = form.cleaned_data['field_ml_model_update_type']
			new_experiment.user_name = form.cleaned_data['user_name']

			print('form params')
			print(new_experiment.field_ml_model_accuracy)
			print(new_experiment.field_ml_model_calibration)
			print(new_experiment.field_ml_model_update_type)
			print(new_experiment.user_name)

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

			return HttpResponseRedirect('/mainpage/')

		else:
			# Create an empty form instance
			form = InitExperimentForm()

			return render(request, 'start_experiment.html', {'form': form})	

	else:

		# Create an empty form instance
		form = InitExperimentForm()

		return render(request, 'start_experiment.html', {'form': form})	

		



	# Render the HTML template index.html with the data in the context variable
	return render(request, 'start_experiment.html', context=context)


def updateButton(request):
	""" View function for asking questions during update """

	# Check if user agreed with AI
	print()


# def index(request):
#     return HttpResponse("Hello, world. You're at the polls index.")