from django.db import models
from django.db import models
import json
import random
import copy
import numpy as np

class Disease:
    def __init__(self):
        self.features = ['cough', 'chills', 'flu_test', 'body_temp', 'weight']
        self.feature_values = {'cough': [0,1], 'chills': [0,1], 'flu_test': [0,1], 
                               'body_temp': ['low', 'norm', 'high'], 'weight': ['low', 'med', 'high']}
        self.cases = [str(self.feature_values['cough'][(i//36)%2]) + "-" + 
                       str(self.feature_values['chills'][(i//18)%2]) + "-" + 
                       str(self.feature_values['flu_test'][(i//9)%2]) + "-" + 
                       self.feature_values['body_temp'][(i//3)%3] + "-" +
                       self.feature_values['weight'][(i)%3] for i in range(2*2*2*3*3)]

# Create your models here.
class ModelMLModel(models.Model):
    domain = Disease()

    # Fields
    accuracy_field = models.TextField(blank=True, null=True, default='{}')
    calibration_field = models.TextField(blank=True, null=True, default='{}')
    update_type_field = models.IntegerField()
    batched_accuracy_field = models.TextField(blank=True, null=True, default='{}')

    # Metadata
    class Meta: 
        ordering = ['-accuracy_field', '-calibration_field', '-update_type_field', '-batched_accuracy_field']

    # Methods
    def initialize(self, performance, calibration, update):

        accuracy = {}
        if (performance == 1): #good model
            for case in self.domain.cases:
                accuracy[case] = 0.75
        else: #performance == 0, poor model
            for case in self.domain.cases:
                accuracy[case] = 0.5
        
        calibration = {} #stores the standard deviation
        if (calibration == 2): #well-calibrated for all
            for case in self.domain.cases:
                calibration[case] = .05
        elif (calibration == 1): #well-calibrated for 2 random features
            high_cal_features = random.sample(self.domain.features[:-1], 2) #choose 2 of the 4 important features
            high_cal_features_vals = [random.sample(self.domain.feature_values[feat], 1) for feat in high_cal_features]
            for case in self.domain.cases:
                case_arr = case.split("-")
                if (case_arr[self.domain.features.index(high_cal_features[0])] == high_cal_features_vals[0] or
                    case_arr[self.domain.features.index(high_cal_features[1])] == high_cal_features_vals[1]):
                    scalibration[case] = .05
                else:
                    calibration[case] = .20
        else: #calibration == 0, not well-calibrated for all
            for case in self.domain.cases:
                calibration[case] = .20
        
        update_type = update
        if(update_type == 0):
            batched_accuracy = copy.deepcopy(accuracy)
        
        self.update_type_field = update_type
        self.accuracy_field = json.dumps(accuracy)
        self.calibration_field = json.dumps(calibration)
        self.batched_accuracy_field = json.dumps(batched_accuracy)
    
    def batch_update(self, case):
        self.accuracy_field = self.batched_accuracy_field   

    def model_update(self, case):
        batched_accuracy = json.loads(self.batched_accuracy_field)

        case_arr = case.split("-")
        #update the particular example
        batched_accuracy[case] = 1.0
        #update related classes
        related = random.sample(self.domain.features, 2)
        for case2 in self.domain.cases:
            case2_arr = case2.split("-")
            if (case_arr[self.domain.features.index(related[0])] == case2_arr[self.domain.features.index(related[0])] or
                case_arr[self.domain.features.index(related[1])] == case2_arr[self.domain.features.index(related[1])]):
                batched_accuracy[case2] += (1-batched_accuracy[case2])*0.15
        
        self.batched_accuracy_field = json.dumps(batched_accuracy)

        if (self.update_type_field == 1): #immediate updates
            self.batch_update(case)
    
    def ground_truth(self, case):
        case_arr = case.split("-")

        if (case_arr[3] == 'high' and (case_arr[0] == 1 or case_arr[1] == 1) and case_arr[2] == 0):
            return 1
        elif (case_arr[3] == 'norm' and case_arr[0] == 1 and case_arr[1] == 1):
            return 1
        return 0

    def model_prediction(self, case):
        gt = self.ground_truth(case)

        accuracy = json.loads(self.accuracy_field)
        calibration = json.loads(self.calibration_field)

        if (random.random <= accuracy[case]):
            prediction = gt
        else:
            prediction = abs(1-gt)
        
        confidence = list(np.random.normal(accuracy[case], calibration[case], 1))[0]
        return (prediction, confidence)

    def generate_patient(self):
    	patient = random.sample(self.domain.cases, 1)[0]
    	patient_arr = patient.split("-")

    	generated_patient = {}
    	for feat in range(len(self.domain.features)):
			generated_patient[self.domain.features[feat]] = patient_arr[feat]
    	return generated_patient

"""
        accuracy = json.loads(self.accuracy_field)
        calibration = json.loads(self.calibration_field)
        batched_accuracy = json.loads(self.batched_accuracy_field)

        self.accuracy_field = json.dumps(accuracy)
        self.calibration_field = json.dumps(calibration)
        self.batched_accuracy_field = json.dumps(batched_accuracy)
"""

class ModelUserResponse(models.Model):
	"""A typical class defining a model, derived from the Model class."""

	# has an id field from models.Model

	# Binary Features: presence of cough, chills, tested positive for flu
	# Categorical: blood pressure (high/med/low)
	# Continuous Features: body temp (>105, 105-95, 95>), weight (normal = 100)



	USER_RESPONSES = (
		('a','The model is typically wrong in this class'),
		('b','The model is generally incorrect'),
		('c','The model displayed low confidence'),
		('d','I was confident I was right based on the current input/info'),
		('e','Other: Free input')

	)

	# linking fields
	field_data_point_string = models.CharField(max_length=20, help_text="Unique string to specify the input feature combo")

	# Fields
	field_ml_accuracy = models.DecimalField(max_digits=4, decimal_places=4, help_text="ML accuracy at time of question")
	field_ml_confidence = models.DecimalField(max_digits=4, decimal_places=4, help_text="ML confidence at time of question")
	field_ml_prediction = models.IntegerField(help_text="Actual ML prediction")
	field_user_prediction = models.IntegerField(help_text="User prediction")
	field_user_did_update = models.IntegerField(help_text="Whether or not user updated the model")

	field_user_disagree_reason_choices = models.CharField(max_length=1, choices=USER_RESPONSES, blank=True, 
				default='m', help_text='If user does not use model prediction, ask why')
	field_user_disagree_reason_freetext = models.TextField(help_text='If user chose "other", provide freetext box')

	# # Metadata
	# class Meta: 
	# 	ordering = ['-my_field_name']

	# Methods
	def get_absolute_url(self):
		"""Returns the url to access a particular instance of MyModelName."""
		return reverse('model-detail-view', args=[str(self.id)])

	def __str__(self):
		"""String for representing the MyModelName object (in Admin site etc.)."""
		return self.my_field_name


