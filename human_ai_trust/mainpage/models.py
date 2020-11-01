from django.db import models

# Create your models here.


class ModelDataPoint(models.Model):

    pass

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
	field_data_point_id = models.ForeignKey('ModelDataPoint', on_delete=models.SET_NULL, null=True)

	# Fields
	field_ml_accuracy = models.DecimalField(max_digits=4, decimal_places=4, "ML accuracy at time of question")
	field_ml_confidence = models.DecimalField(max_digits=4, decimal_places=4, help_text="ML confidence at time of question")
	field_ml_prediction = models.BinaryField(help_text="Actual ML prediction")
	field_user_prediction = models.BinaryField(help_text="User prediction")
	field_user_did_update = models.BinaryField(help_text="Whether or not user updated the model")

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