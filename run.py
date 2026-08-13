"""
run.py — Launcher cepat untuk aplikasi Analisis Sentimen Dakwah.

Cara pakai:
    python run.py
"""

import subprocess
import sys
import os

# Pastikan working directory adalah root proyek
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Cari executable streamlit: utamakan yang ada di .venv
STREAMLIT_VENV = os.path.join(".venv", "Scripts", "streamlit.exe")
STREAMLIT_FALLBACK = "streamlit"

streamlit_cmd = STREAMLIT_VENV if os.path.exists(STREAMLIT_VENV) else STREAMLIT_FALLBACK

cmd = [streamlit_cmd, "run", "app.py"]

print("Memulai Sistem Analisis Sentimen Konten Dakwah Digital...")
print(f"   Buka browser di: http://localhost:8501")
print("   Tekan Ctrl+C untuk menghentikan.\n")

try:
    subprocess.run(cmd, check=True)
except KeyboardInterrupt:
    print("\nAplikasi dihentikan.")
except FileNotFoundError:
    print("Streamlit tidak ditemukan. Jalankan: pip install -r requirements.txt")
    sys.exit(1)
