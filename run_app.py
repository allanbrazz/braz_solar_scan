# run_app.py
import os, socket, webbrowser, sys
from pathlib import Path

# (opcional, mas ajuda o PyInstaller a achar seu pacote do projeto)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Força o Django a rodar sem debug no executável, a menos que você passe DJANGO_DEBUG=1
os.environ.setdefault("DJANGO_DEBUG", "0")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")  # ajuste se seu pacote não for "config"

# >>> IMPORTANTE: importar django ANTES de chamar django.setup()
import django
django.setup()

from django.core.management import call_command
from django.core.wsgi import get_wsgi_application
from waitress import serve

def _find_free_port(start=8000, limit=50):
    port = start
    for _ in range(limit):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    return start

def main():
    # Migrações automáticas no primeiro start
    try:
        call_command("migrate", interactive=False, verbosity=0)
    except Exception as e:
        print("Aviso: migrate falhou:", e)

    application = get_wsgi_application()
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}/"
    try:
        webbrowser.open(url)
    except Exception:
        pass

    print(f"SolarControl rodando em {url}")
    serve(application, listen=f"127.0.0.1:{port}")

if __name__ == "__main__":
    main()
