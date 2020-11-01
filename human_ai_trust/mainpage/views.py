from django.shortcuts import render

from mainpage.models import ModelUserResponse

# Create your views here.

from django.http import HttpResponse

def index(request):
	"""View function for home page of site."""

	new_user_response = ModelUserResponse(field_data_point_string='0-0-0-low-low',
							field_ml_accuracy = 0.5,
							field_ml_confidence = 0.5,
							field_ml_prediction = 0,
							field_user_prediction = 1,
							field_user_did_update = 1,
							field_user_disagree_reason_choices = 'a',
							field_user_disagree_reason_freetext = 'Example long text'
							)

	new_user_response.save()

	# Generate counts of some of the main objects
	all_responses = ModelUserResponse.objects.all()
	print(len(all_responses))
	num_responses = len(all_responses)


	context = {
	    'user_responses': all_responses,
	    'num_responses': num_responses
	}

	# Render the HTML template index.html with the data in the context variable
	return render(request, 'index.html', context=context)


# def index(request):
#     return HttpResponse("Hello, world. You're at the polls index.")