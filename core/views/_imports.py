# core/views/imports.py
from __future__ import annotations

# =========================
# Standard library
# =========================
import csv
import datetime as dt
import logging
import math
import os
import re
from datetime import date, time
from decimal import Decimal, InvalidOperation
from typing import Any, Optional, TYPE_CHECKING

# =========================
# Third-party
# =========================
import pandas as pd
from dateutil import parser as dtparser

# =========================
# Django
# =========================
from django.conf import settings
from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import Q
from django.http import (
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
    HttpResponseRedirect,
    JsonResponse,
)
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse, reverse_lazy
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.http import require_GET, require_http_methods, require_POST
from django.views.generic import CreateView, DetailView, FormView, ListView, UpdateView



# Helpers
from ._helpers import LOCAL_TZ, _get_float, _to_decimal

# =========================
# Module globals
# =========================
logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    pass

# =========================
# Star-import control
# =========================
# Exporta tudo que NÃO começa com "_" + adiciona explicitamente os helpers com "_"
__all__ = [k for k in globals().keys() if not k.startswith("_")]
__all__ += ["_get_float", "_to_decimal"]
