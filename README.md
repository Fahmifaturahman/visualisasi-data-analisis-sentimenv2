# Sistem Analisis Sentimen Konten Dakwah Digital — Kota Bandung

Aplikasi web berbasis **Streamlit** yang menampilkan seluruh hasil pipeline analisis sentimen komentar YouTube dari konten dakwah Islam penceramah di Kota Bandung.

Dibangun sebagai bagian dari penelitian skripsi.

---

## Prasyarat

- **Python 3.10+** (direkomendasikan)
- Semua dependensi tercantum di `requirements.txt`

---

## Instalasi

```bash
pip install -r requirements.txt
```

---

## Cara Menjalankan

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`.

---

## Tata Letak Direktori

Sebelum menjalankan, pastikan file data dan model tersedia:

```
project_root/
├── app.py
├── requirements.txt
├── patch_cell12.py
├── data/
│   ├── dataset_final_komentar.csv
│   ├── dataset_final_video.csv
│   └── dataset_final_per_penceramah.csv
├── model/
│   ├── model_naive_bayes.pkl
│   └── tfidf_vectorizer.pkl
├── eda_report/              ← Opsional: 9 file PNG chart dari notebook
└── tests/
    ├── test_unit.py
    └── test_properties.py
```

---

## Lima Halaman Aplikasi

| Halaman | Deskripsi |
|---------|-----------|
| **Overview** | Ringkasan penelitian, diagram pipeline, dan statistik utama dataset |
| **EDA Dashboard** | Visualisasi interaktif Plotly dan word cloud dengan filter penceramah/topik |
| **Evaluasi Model** | Confusion matrix, metrik akurasi/precision/recall/F1, dan tabel misklasiﬁkasi |
| **Demo Klasifikasi Real-Time** | Klasifikasi komentar baru dengan langkah preprocessing step-by-step |
| **Pencarian Penceramah & Sentimen Video** | Profil penceramah, daftar video, sentimen per video, dan embed YouTube |

---

## Menjalankan Tests

```bash
# Unit tests
pytest tests/test_unit.py -v

# Property-based tests (Hypothesis)
pytest tests/test_properties.py -v
```

---

## Catatan: Patch Pipeline Colab

Sebelum mengekspor file CSV dari Google Colab, terapkan patch `patch_cell12.py` ke **Cell 12 pada Script 3** Colab. Patch ini menambahkan kolom `video_id` ke semua file CSV output sehingga fitur pencarian berdasarkan video dapat berfungsi dengan benar.

Lihat komentar di dalam `patch_cell12.py` untuk instruksi lengkap.

---

## Mengubah Identitas Peneliti

Buka `app.py` dan cari baris:

```python
# --- EDIT SESUAI DATA PENELITI ---
NAMA_PENELITI = "Nama Lengkap"
NIM           = "XXXXXXXXXX"
```

Ganti dengan nama dan NIM Anda.
