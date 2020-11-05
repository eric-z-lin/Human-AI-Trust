from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('start_experiment', views.start_experiment, name='start_experiment'),
    path('patient_result', views.patient_result, name='patient_result'),
]
