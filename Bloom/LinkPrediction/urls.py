from django.urls import path
from . import views

urlpatterns = [
    path('link_prediction/', views.handle_linkpred_req, name='link_prediction'),
]