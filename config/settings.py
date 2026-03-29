"""
Django settings for config project.

Configuração ajustada para:
- desenvolvimento local;
- execução empacotada via PyInstaller/Waitress;
- uso de variáveis de ambiente sem hardcodes sensíveis no repositório.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv

from .runtime_paths import runtime_base_dir


# -----------------------------------------------------------------------------
# Paths e helpers de ambiente
# -----------------------------------------------------------------------------
SOURCE_BASE_DIR = Path(__file__).resolve().parent.parent
RUNTIME_BASE_DIR = runtime_base_dir()
BASE_DIR = RUNTIME_BASE_DIR

# Carrega .env da raiz do projeto quando existir.
# Em execução "frozen", SOURCE_BASE_DIR continua apontando para a raiz do código.
load_dotenv(SOURCE_BASE_DIR / ".env")



def env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value != "" else default



def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}



def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ImproperlyConfigured(f"{name} deve ser inteiro.") from exc



def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ImproperlyConfigured(f"{name} deve ser float.") from exc



def env_list(name: str, default: list[str] | None = None, sep: str = ",") -> list[str]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return list(default or [])
    return [item.strip() for item in value.split(sep) if item.strip()]



def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"))



def _user_data_dir() -> Path:
    home = Path.home()
    if os.name == "nt":
        base = Path(os.getenv("APPDATA", home / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = home / "Library" / "Application Support"
    else:
        base = Path(os.getenv("XDG_DATA_HOME", home / ".local" / "share"))
    data_dir = base / "SolarControl"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


APP_ENV = (env_str("DJANGO_ENV", "dev") or "dev").lower()
DEBUG = env_bool("DJANGO_DEBUG", default=(APP_ENV != "prod"))

_secret_key = env_str("DJANGO_SECRET_KEY")
if _secret_key:
    SECRET_KEY = _secret_key
elif DEBUG or _is_frozen():
    # Fallback legado para preservar compatibilidade com sessões e ambiente local existente.
    # Em produção, defina DJANGO_SECRET_KEY explicitamente.
    SECRET_KEY = "django-insecure-@8goz41^$w#icx*3wr6%q-)uzjc0jcg7fk75(15ji_k89g*nz*"
else:
    raise ImproperlyConfigured("Defina DJANGO_SECRET_KEY para executar com DEBUG=False.")

_default_allowed_hosts = ["127.0.0.1", "localhost"]
ALLOWED_HOSTS = env_list(
    "DJANGO_ALLOWED_HOSTS",
    default=_default_allowed_hosts if DEBUG or _is_frozen() else [],
)

CSRF_TRUSTED_ORIGINS = env_list("DJANGO_CSRF_TRUSTED_ORIGINS", default=[])


# -----------------------------------------------------------------------------
# Application definition
# -----------------------------------------------------------------------------
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "whitenoise.runserver_nostatic",
    "django.contrib.staticfiles",
    "core",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [SOURCE_BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"


# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------
DB_ENGINE = env_str("DJANGO_DB_ENGINE", "django.db.backends.sqlite3")
DB_NAME = env_str("DJANGO_DB_NAME")

if DB_ENGINE == "django.db.backends.sqlite3":
    if DB_NAME:
        db_name = Path(DB_NAME).expanduser()
    else:
        # Mantém o caminho legado do projeto para não trocar o banco usado no ambiente local.
        db_name = _user_data_dir() / "solar.sqlite3"

    DATABASES = {
        "default": {
            "ENGINE": DB_ENGINE,
            "NAME": db_name,
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": DB_ENGINE,
            "NAME": env_str("DJANGO_DB_NAME", ""),
            "USER": env_str("DJANGO_DB_USER", ""),
            "PASSWORD": env_str("DJANGO_DB_PASSWORD", ""),
            "HOST": env_str("DJANGO_DB_HOST", ""),
            "PORT": env_str("DJANGO_DB_PORT", ""),
        }
    }


# -----------------------------------------------------------------------------
# Password validation
# -----------------------------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]


# -----------------------------------------------------------------------------
# Internationalization
# -----------------------------------------------------------------------------
LANGUAGE_CODE = env_str("DJANGO_LANGUAGE_CODE", "pt-br")
TIME_ZONE = env_str("DJANGO_TIME_ZONE", "UTC")
USE_I18N = True
USE_TZ = True


# -----------------------------------------------------------------------------
# Static / media
# -----------------------------------------------------------------------------
STATIC_URL = "/static/"
STATIC_ROOT = Path(env_str("DJANGO_STATIC_ROOT", str(SOURCE_BASE_DIR / "staticfiles")))
STATICFILES_DIRS = [SOURCE_BASE_DIR / "static"] if (SOURCE_BASE_DIR / "static").exists() else []

MEDIA_URL = "/media/"
MEDIA_ROOT = Path(env_str("DJANGO_MEDIA_ROOT", str(SOURCE_BASE_DIR / "media")))

STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": (
            "django.contrib.staticfiles.storage.StaticFilesStorage"
            if DEBUG
            else "whitenoise.storage.CompressedManifestStaticFilesStorage"
        )
    },
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# -----------------------------------------------------------------------------
# Auth / e-mail
# -----------------------------------------------------------------------------
LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "home"
LOGOUT_REDIRECT_URL = "login"

EMAIL_BACKEND = env_str(
    "DJANGO_EMAIL_BACKEND",
    "django.core.mail.backends.console.EmailBackend",
)
DEFAULT_FROM_EMAIL = env_str("DEFAULT_FROM_EMAIL", "no-reply@example.com")


# -----------------------------------------------------------------------------
# Integrações externas
# -----------------------------------------------------------------------------
# NREL / NSRDB
NREL_API_KEY = env_str("NREL_API_KEY") or env_str("NSRDB_API_KEY", "")
NREL_FULL_NAME = env_str("NREL_FULL_NAME", "Allan Braz")
NREL_EMAIL = env_str("NREL_EMAIL", "allan.dasilva@utec.edu.uy")
NREL_AFFILIATION = env_str("NREL_AFFILIATION", "UTEC")
NREL_REASON = env_str("NREL_REASON", "research")
NREL_MAILING_LIST = env_str("NREL_MAILING_LIST", "false")

# Renovigi / ShineMonitor
RENOVIGI_BASE_URL = env_str("RENOVIGI_BASE_URL", "https://web.shinemonitor.com/public/")
RENOVIGI_COMPANY_KEY = env_str("RENOVIGI_COMPANY_KEY", "")
RENOVIGI_HTTP_TIMEOUT = env_float("RENOVIGI_HTTP_TIMEOUT", 30.0)
RENOVIGI_PAGE_SIZE = env_int("RENOVIGI_PAGE_SIZE", 200)
RENOVIGI_SLEEP_SEC = env_float("RENOVIGI_SLEEP_SEC", 0.3)
RENOVIGI_MAX_RETRIES = env_int("RENOVIGI_MAX_RETRIES", 5)

# Aliases para compatibilidade com código legado
SHINEMONITOR_BASE_URL = RENOVIGI_BASE_URL
SHINEMONITOR_HTTP_TIMEOUT = RENOVIGI_HTTP_TIMEOUT


# -----------------------------------------------------------------------------
# Diretórios de dados do produto
# -----------------------------------------------------------------------------
APP_DATA_DIR = Path(env_str("SOLARCONTROL_DATA_DIR", str(_user_data_dir()))).expanduser()
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

IV_FDD_DIR = APP_DATA_DIR / "iv_fdd"
IV_FDD_DATASETS_DIR = IV_FDD_DIR / "datasets"
IV_FDD_MODELS_DIR = IV_FDD_DIR / "models"

MPPT_GNN_FDD_DIR = APP_DATA_DIR / "mppt_gnn_fdd"
MPPT_GNN_FDD_MODELS_DIR = MPPT_GNN_FDD_DIR / "models"

for _path in (
    APP_DATA_DIR,
    IV_FDD_DIR,
    IV_FDD_DATASETS_DIR,
    IV_FDD_MODELS_DIR,
    MPPT_GNN_FDD_DIR,
    MPPT_GNN_FDD_MODELS_DIR,
    STATIC_ROOT,
    MEDIA_ROOT,
):
    _path.mkdir(parents=True, exist_ok=True)
