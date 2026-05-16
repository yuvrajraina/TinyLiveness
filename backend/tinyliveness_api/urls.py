from __future__ import annotations

from django.http import JsonResponse
from django.urls import include, path


def health(_: object) -> JsonResponse:
    return JsonResponse({"status": "ok", "service": "tinyliveness"})


urlpatterns = [
    path("api/liveness/", include("liveness.urls")),
    path("health/", health),
]
