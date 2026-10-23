# Sistem Analisis Sentimen Konten Dakwah Digital — Kota Bandung

Aplikasi web berbasis **Streamlit** yang menyajikan hasil pipeline analisis sentimen komentar YouTube dari konten dakwah Islam penceramah di Kota Bandung.

Dibangun sebagai bagian dari penelitian skripsi.

---

## Prasyarat

- **Python 3.10+**
- Semua dependensi tercantum di `requirements.txt`

---

## Instalasi

```bash
pip install -r requirements.txt
```

Untuk development dan testing:

```bash
pip install -r requirements-dev.txt
```

---

## Cara Menjalankan

```bash
streamlit run app.py
```

Atau gunakan launcher:

```bash
python run.py
# atau
run.bat
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`.

---

## Struktur Direktori

```
project_root/
├── app.py                          ← Aplikasi Streamlit utama
├── run.py                          ← Launcher Python
├── run.bat                         ← Launcher Windows
├── requirements.txt
├── README.md
│
├── data/
│   ├── hasil_prediksi_lengkap.csv          ← Komentar + label + prediksi (sumber utama)
│   ├── dataset_final_komentar.csv          ← Dataset komentar final
│   ├── dataset_final_video.csv             ← Metadata video + sentimen agregat
│   ├── dataset_final_per_penceramah.csv    ← Statistik sentimen per penceramah
│   ├── ringkasan_hasil.json                ← Ringkasan performa model
│   └── raw/
│       ├── raw_data_komentar.csv
│       ├── raw_data_video.csv
│       ├── clean_data_komentar.csv
│       ├── clean_data_stemmed.csv
│       ├── clean_data_video.csv
│       ├── komentar_tidak_berlabel.csv
│       └── progress_checkpoint.json
│
├── model/
│   ├── model_naive_bayes.pkl       ← Model Multinomial Naive Bayes terlatih
│   └── tfidf_vectorizer.pkl        ← TF-IDF Vectorizer terlatih
│
├── eda_report/                     ← Grafik PNG hasil notebook
│   ├── eda01_sebelum_sesudah.png
│   ├── eda02_distribusi_panjang.png
│   ├── eda03_top_penceramah.png
│   ├── eda04_distribusi_topik.png
│   ├── eda05_tren_bulanan.png
│   ├── eda06_wordcloud.png
│   ├── eda07_top30_kata.png
│   ├── eda08_heatmap_penceramah_topik.png
│   ├── eda09_word2vec_pca.png
│   ├── eda_summary.txt
│   ├── grafik_confusion_matrix_normalized.png
│   ├── grafik_distribusi_label.png
│   └── grafik_evaluasi_naive_bayes.png
│
├── notebooks/
│   ├── Script1_Scraping_Dakwah_Bandung.ipynb
│   ├── Script2_Cleaning_EDA_v3_word2vec.ipynb
│   ├── Script2_Cleaning_EDA_v3_word2vec_alay_v2.ipynb   ← Versi terbaru
│   ├── Script3_v3_DatasetFinal.ipynb
│   ├── Script3_v3_DatasetFinal_binary_lexicon.ipynb     ← Versi terbaru
│   └── patch_cell12.py                                  ← Patch video_id untuk Colab
│
└── tests/
    ├── __init__.py
    ├── test_unit.py
    └── test_properties.py
```

---

## Enam Halaman Aplikasi

| Halaman | Deskripsi |
|---------|-----------|
| **Overview** | Ringkasan penelitian, pipeline analisis, KPI dataset & model |
| **EDA Dashboard** | Visualisasi interaktif (sentimen, penceramah, topik, teks, word cloud) |
| **Evaluasi Model** | Confusion matrix, metrik akurasi/precision/recall/F1, classification report |
| **Demo Klasifikasi Real-Time** | Klasifikasi komentar baru dengan preprocessing step-by-step |
| **Pencarian Penceramah & Video** | Profil penceramah, sentimen per video, embed YouTube |
| **Ranking Penceramah** | Peringkat penceramah berdasarkan persentase sentimen positif |

---

## Menjalankan Tests

```bash
# Install dependensi dev dulu
pip install -r requirements-dev.txt

# Unit tests
pytest tests/test_unit.py -v

# Property-based tests (Hypothesis)
pytest tests/test_properties.py -v
```

---

## Deploy ke Streamlit Cloud

1. Push repo ke GitHub (pastikan semua file di tabel bawah sudah ter-commit)
2. Buka [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Pilih repo → branch `main` → Main file: `app.py`
4. Klik **Deploy**

**File yang wajib ada di repo untuk deploy berhasil:**

| File | Keterangan |
|------|-----------|
| `app.py` | Aplikasi utama |
| `requirements.txt` | Dependensi Python |
| `data/hasil_prediksi_lengkap.csv` | Data komentar utama |
| `data/dataset_final_video.csv` | Metadata video |
| `data/dataset_final_per_penceramah.csv` | Statistik penceramah |
| `data/ringkasan_hasil.json` | Ringkasan model |
| `model/model_naive_bayes.pkl` | Model MNB |
| `model/tfidf_vectorizer.pkl` | TF-IDF Vectorizer |

---

## Mengubah Identitas Peneliti

Buka `app.py` dan ubah:

```python
NAMA_PENELITI = "Fahmi Faturahman"
NIM           = "1222008"
```

---

## Catatan Notebook

Seluruh notebook tersimpan di folder `notebooks/`. Versi terbaru (dengan kamus alay dan klasifikasi binary):
- `Script2_Cleaning_EDA_v3_word2vec_alay_v2.ipynb`
- `Script3_v3_DatasetFinal_binary_lexicon.ipynb`

File `patch_cell12.py` berisi instruksi patch `video_id` untuk Cell 12 di Script 3 Colab jika diperlukan di iterasi berikutnya.
