from django.urls import path
from . import views

urlpatterns = [
    path('link_prediction/', views.link_prediction, name='link_prediction'),
]