from __future__ import annotations

from django.urls import path

from .views import model_info, predict

urlpatterns = [
    path("model/", model_info, name="liveness_model_info"),
    path("predict/", predict, name="liveness_predict"),
]
