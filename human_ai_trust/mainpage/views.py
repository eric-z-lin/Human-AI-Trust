from django.shortcuts import render, redirect
from django import forms

from mainpage.models import *

# Create your views here.

from django.http import HttpResponse, HttpResponseRedirect
import csv

CONST_BATCH_UPDATE_FREQUENCY = 2
MAX_TRIALS = 50



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

		# Need to save here since generate_patient() updates the field_patient_number
		experiment.save()
		
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

		request.session['user_response_id'] = new_user_response.id
		print('index_ur_id', new_user_response.id)

		# Build forms
		trustForm = UserTrustForm()
		updateForm = UpdateForm()


		context = {
		    # 'feature_dict': feature_dict,
		    'ml_model_prediction': ('Negative' if model_prediction == 0 else 'Positive'),
		    'ml_confidence': str(model_confidence) + "%",
		    'feature_display_dict': feature_display_dict,
		    'user_response': new_user_response,
		    'form1': UserTrustForm(),
		    'form2': UpdateForm(),
		    'score': experiment.field_score,
		    'patient_num': experiment.field_patient_number,
		    'MAX_TRIALS': MAX_TRIALS,
		    'percent_diagnosed': round(experiment.field_patient_number * 100 / MAX_TRIALS),
		    'name':experiment.field_user_name,
		    'ground_truth': new_user_response.field_instance_ground_truth,
		}

		# Render the HTML template index.html with the data in the context variable
		return render(request, 'index.html', context=context)



class InitExperimentForm(forms.Form):
	user_name = forms.CharField(label='Your name', max_length=100)
	ACCURACY_CHOICES =( 
		(0, "Giraffe"),
		(1, "Lion")
	)
	field_ml_model_accuracy = forms.ChoiceField(choices = ACCURACY_CHOICES)

	CALIBRATION_CHOICES =( 
		(0, "Pizza"),
		(1, "Bagel"),
		(2, "Pho")
	)
	field_ml_model_calibration = forms.ChoiceField(choices = CALIBRATION_CHOICES)

	UPDATE_TYPE_CHOICES =( 
		(0, "Rome"),
		(1, "Geneva"),
		(2, "London")
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
			new_experiment.field_user_name = form.cleaned_data['user_name']

			print('form params')
			print(new_experiment.field_ml_model_accuracy)
			print(new_experiment.field_ml_model_calibration)
			print(new_experiment.field_ml_model_update_type)
			print(new_experiment.field_user_name)

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
			request.session['batch_update_requested'] = False
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


class UpdateForm(forms.Form):
	TRUST_CHOICES = (
		(0, "Not at all"),
		(1, "Not really"),
		(2, "Indifferent"),
		(3, "A little"),
		(4, "A lot"),
	)
	field_trust = forms.ChoiceField(label="How much do you trust the AI?", choices = TRUST_CHOICES, widget=forms.RadioSelect)
	field_disagreement_choice = forms.ChoiceField(label="Why do you disagree with the AI?", choices = ModelUserResponse.USER_DISAGREE_REASON_RESPONSES, widget=forms.RadioSelect)
	field_disagreement_text = forms.CharField(label='Reason for disagreement', max_length=160)


class UserTrustForm(forms.Form):
	TRUST_CHOICES = (
		(0, "Not at all"),
		(1, "Not really"),
		(2, "Indifferent"),
		(3, "A little"),
		(4, "A lot"),
	)
	field_trust = forms.ChoiceField(label="How much do you trust the AI?", choices = TRUST_CHOICES, widget=forms.RadioSelect)

def patient_result(request):
	""" View function for updating user_response and generating patient result page """

	# Query for model, experiment case, user response
	# Load session Id
	user_response_id = request.session['user_response_id']
	user_response = ModelUserResponse.objects.get(id=user_response_id)
	experiment = user_response.field_experiment
	ml_model = experiment.field_model_ml_model


	# Table for patient case
	domain = ml_model.domain
	feature_display_dict = {}
	case_values = user_response.field_data_point_string.split('-')
	for feat, case_value in zip(domain.features, case_values):
		feature_display_dict[domain.feature_names[feat]] = domain.feature_value_names[feat + "-" + case_value]


	if request.POST.get("next-trial"):
		if experiment.field_patient_number >= MAX_TRIALS:
			return HttpResponseRedirect('/mainpage/complete')

		else:
			return HttpResponseRedirect('/mainpage/')



	ml_prediction = user_response.field_ml_prediction

	print("what is this literally?")
	print(request.POST.get("agree"))
	print(request.POST.get("disagree-no-update"))
	print(request.POST.get("disagree-update"))

	# Check which button got pressed
	if request.POST.get("agree"):
		print('reached AGREE')
		# Create a form instance with the submitted data
		form = UserTrustForm(request.POST, initial={'field_trust': 2})  # 2
		# Validate the form
		print('checking validity')
		if form.is_valid(): 

			# Instantiate models
			user_response.field_user_trust = form.cleaned_data['field_trust']
			user_response.field_user_prediction = ml_prediction


	# Check which button got pressed
	if request.POST.get("disagree-no-update"):
		# Create a form instance with the submitted data
		print('reached disagree-no-update')
		form = UpdateForm(request.POST, initial={'field_trust': 2})  # 2
		# Validate the form
		print('checking validity')
		if form.is_valid(): 

			# Instantiate models
			user_response.field_user_trust = form.cleaned_data['field_trust']
			user_response.field_user_prediction = abs(1 - ml_prediction)
			user_response.field_user_disagree_reason_choices = form.cleaned_data['field_disagreement_choice']
			user_response.field_user_disagree_reason_freetext = form.cleaned_data['field_disagreement_text']


	update_bool = False
		# Check which button got pressed
	if request.POST.get("disagree-update"):
		# Create a form instance with the submitted data
		print('reached disagree-UPDATE')
		form = UpdateForm(request.POST, initial={'field_trust': 2})  # 2
		# Validate the form
		print('checking validity')
		if form.is_valid(): 

			# Instantiate models
			user_response.field_user_trust = form.cleaned_data['field_trust']
			user_response.field_user_prediction = abs(1 - ml_prediction)
			user_response.field_user_disagree_reason_choices = form.cleaned_data['field_disagreement_choice']
			user_response.field_user_disagree_reason_freetext = form.cleaned_data['field_disagreement_text']

			ml_model.model_update(user_response.field_data_point_string, 
				user_response.field_user_prediction, user_response.field_instance_ground_truth)

			if experiment.field_ml_model_update_type == 1:
				update_bool = True

			request.session['batch_update_requested'] = True

	# if updatetype is 1, it'll handle updating on its own
	if experiment.field_ml_model_update_type == 2 and experiment.field_patient_number%CONST_BATCH_UPDATE_FREQUENCY == 0:
		if request.session['batch_update_requested']:
			update_bool = True
			request.session['batch_update_requested'] = False
		ml_model.batch_update()



	print('user_response_id', user_response_id)
	print('user', user_response.field_user_prediction)
	print('ml_prediction', ml_prediction)
	print('ground truth',user_response.field_instance_ground_truth)

	# Set scores
	score_update = 0
	if user_response.field_instance_ground_truth == user_response.field_user_prediction:
		experiment.field_score += 2
		score_update = 2
	else:
		experiment.field_score += -4
		score_update = -4


	user_response.save()
	ml_model.save()
	experiment.save()



	# Write user response to a csv
	write_to_csv(user_response)

	# Render patient result page

	context = {
		'feature_display_dict': feature_display_dict,
		'ml_prediction': 'Positive' if (ml_prediction == 1) else 'Negative',
		'user_prediction': 'Positive' if (user_response.field_user_prediction == 1) else 'Negative',
		'ground_truth': 'Positive' if (user_response.field_instance_ground_truth == 1) else 'Negative',
		'score_update': score_update,
		'field_score': experiment.field_score,
		'result_color': 'lightgreen' if (user_response.field_user_prediction == user_response.field_instance_ground_truth) else 'lightcoral',
		'update_bool': update_bool,
	}

	return render(request, 'patient_result.html', context=context)



def experiment_complete(request):
	experiment_id = request.session['experiment_id']
	experiment = ModelExperiment.objects.get(id=experiment_id)

	context = {
		'score':experiment.field_score,
		'name':experiment.field_user_name
	}


	# Render the HTML template index.html with the data in the context variable
	return render(request, 'complete.html', context=context)


def write_to_csv(user_response):
	fields = [user_response.field_experiment.field_patient_number, user_response.field_data_point_string,
					user_response.field_ml_accuracy, user_response.field_ml_calibration,
					user_response.field_ml_prediction,
					user_response.field_instance_ground_truth, user_response.field_user_prediction,
					user_response.field_user_did_update, user_response.field_user_disagree_reason_choices, 
					user_response.field_user_disagree_reason_freetext, user_response.field_user_trust]


	with open('experiments/experiment-'+str(user_response.field_experiment.id)+'.csv','a') as f:
		writer = csv.writer(f)
		writer.writerow(fields)


# def index(request):
#     return HttpResponse("Hello, world. You're at the polls index.")