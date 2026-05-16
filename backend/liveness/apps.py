from __future__ import annotations

from django.apps import AppConfig


class LivenessConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "liveness"
