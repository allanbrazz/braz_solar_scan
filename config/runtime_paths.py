# config/runtime_paths.py
from pathlib import Path
import sys

def runtime_base_dir() -> Path:
    # Quando "congelado" pelo PyInstaller (onefile), os assets são extraídos em _MEIPASS
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS)  # raiz extraída em runtime
    # Em desenvolvimento: pasta do projeto (onde fica manage.py/config/etc.)
    return Path(__file__).resolve().parent.parent