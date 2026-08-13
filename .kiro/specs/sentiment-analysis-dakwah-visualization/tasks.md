# Implementation Plan: Sistem Analisis Sentimen Konten Dakwah Digital — Kota Bandung

## Overview

Aplikasi web Streamlit monolitik (`app.py`) yang membaca artefak statis (CSV + model `.pkl`) dari pipeline Colab dan menyajikannya dalam lima halaman interaktif. Implementasi mengikuti urutan dependensi: setup → konstanta → cache layer → preprocessing → UI helpers → halaman-halaman → error handling → patch data → testing. Notebook Colab disimpan di direktori `notebooks/` dan data mentah/intermediate di `data/raw/`.

Bahasa: **Python 3.x + Streamlit**

---

## Tasks

- [x] 1. Setup struktur proyek dan dependensi
  - Buat file `requirements.txt` di root proyek dengan semua dependensi versi eksplisit: `streamlit==1.35.0`, `pandas==2.2.2`, `numpy==1.26.4`, `plotly==5.22.0`, `Pillow==10.3.0`, `PySastrawi==1.2.0`, `scikit-learn==1.5.0`, `wordcloud==1.9.3`, `matplotlib==3.9.0`, `hypothesis==6.104.2`, `pytest==8.2.2`
  - Buat direktori `data/`, `data/raw/`, `model/`, `eda_report/`, `notebooks/`
  - Buat direktori `tests/` dengan file `tests/__init__.py` kosong
  - _Requirements: 8.1, 8.2, 8.3_

- [x] 2. Buat `app.py` skeleton dengan konfigurasi halaman dan routing sidebar
  - Buat file `app.py` di root proyek
  - Tambahkan komentar header di bagian atas yang menjelaskan: cara instalasi (`pip install -r requirements.txt`), cara menjalankan (`streamlit run app.py`), tata letak direktori yang diperlukan (`data/`, `model/`, `eda_report/`), dan instruksi mengedit `NAMA_PENELITI`/`NIM`
  - Panggil `st.set_page_config(page_title="Sistem Analisis Sentimen Konten Dakwah Digital — Kota Bandung", layout="wide", page_icon="🕌")` sebagai perintah pertama setelah import
  - Tambahkan sidebar navigation menggunakan `st.sidebar.radio` dengan lima opsi: `"Overview"`, `"EDA Dashboard"`, `"Evaluasi Model"`, `"Demo Klasifikasi Real-Time"`, `"Pencarian Penceramah & Sentimen Video"`
  - Tambahkan stub kosong untuk setiap fungsi halaman: `page_overview()`, `page_eda()`, `page_evaluasi()`, `page_demo()`, `page_pencarian()` — masing-masing berisi `st.write("TODO")` sementara
  - Tambahkan blok `if __name__ == "__main__"` atau kode routing utama di bawah: baca pilihan sidebar, dispatch ke fungsi halaman yang sesuai
  - _Requirements: 1.1, 1.2, 1.7, 8.2_

- [x] 3. Definisikan konstanta global di `app.py`
  - Tambahkan konstanta identitas peneliti dengan komentar `# EDIT SESUAI DATA PENELITI`: `NAMA_PENELITI = "Nama Lengkap"` dan `NIM = "XXXXXXXXXX"`
  - Definisikan `WARNA_SENTIMEN = {"Positif": "#2ecc71", "Negatif": "#e74c3c", "Netral": "#95a5a6"}`
  - Definisikan path constants (relatif terhadap root proyek): `PATH_CSV_KOMENTAR`, `PATH_CSV_VIDEO`, `PATH_CSV_PENCERAMAH`, `PATH_MODEL_NB`, `PATH_VECTORIZER`, `DIR_EDA_REPORT`
  - Definisikan nama kolom kanonik: `KOLOM_LABEL = "sentiment_label"`, `KOLOM_PREDIKSI = "label_prediksi"`, `KOLOM_SPLIT = "split"`, `KOLOM_CHANNEL = "channel_name"`, `KOLOM_TOPIK = "topik_utama"`, `KOLOM_VIDEO_ID = "video_id"`, `KOLOM_KOMENTAR_BERSIH = "komentar_bersih"`
  - _Requirements: 1.3, 8.3_

- [x] 4. Implementasikan fungsi cache `load_data()`, `load_model()`, `init_sastrawi()`
  - Implementasikan `load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]` dengan dekorator `@st.cache_data`: baca tiga CSV menggunakan `pandas.read_csv`; jika file tidak ditemukan raise `FileNotFoundError(f"File tidak ditemukan: {os.path.abspath(path)}")` dengan path absolut
  - Implementasikan `load_model() -> tuple` dengan dekorator `@st.cache_resource`: load `model_naive_bayes.pkl` dan `tfidf_vectorizer.pkl` menggunakan `pickle.load()`; tangani `FileNotFoundError` dan exception umum
  - Implementasikan `init_sastrawi() -> tuple` dengan dekorator `@st.cache_resource`: inisialisasi `StemmerFactory().create_stemmer()` dan `StopWordRemoverFactory().create_stop_word_remover()`; tangani `ImportError` jika PySastrawi tidak terinstall
  - _Requirements: 1.4, 1.5, 1.6, 5.7_

- [x] 5. Implementasikan preprocessing pipeline
  - Implementasikan enam fungsi helper: `hapus_url(t)` menggunakan `re.sub(r'http\S+|www\.\S+', ' ', t)`, `hapus_mention(t)` menggunakan `re.sub(r'[@#]\w+', ' ', t)`, `hapus_emoji(t)` menggunakan regex Unicode emoji, `hapus_nonalpha(t)` menggunakan `re.sub(r'[^a-zA-Z\s]', ' ', t)`, `case_folding(t)` menggunakan `t.lower()`, `norm_spasi(t)` menggunakan `re.sub(r'\s+', ' ', t).strip()`
  - Implementasikan `cleaning_pipeline(teks: str) -> str`: jalankan secara berurutan `hapus_url` → `hapus_mention` → `hapus_emoji` → `hapus_nonalpha` → `case_folding` → `norm_spasi`; output hanya boleh mengandung `[a-z ]` dan tidak ada leading/trailing whitespace
  - Implementasikan `full_preprocessing(teks: str, stopword_remover, stemmer) -> str`: panggil `cleaning_pipeline()` → `stopword_remover.remove()` → `stemmer.stem()` secara berurutan; pipeline harus identik dengan Script 2 dan Script 3 Colab
  - _Requirements: 5.3, 5.8_

- [x] 6. Implementasikan helper UI `get_sentiment_color()` dan `get_sentiment_badge_html()`
  - Implementasikan `get_sentiment_color(sentimen: str) -> str`: lookup `WARNA_SENTIMEN.get(sentimen, "#95a5a6")`; kembalikan default abu-abu jika label tidak dikenali
  - Implementasikan `get_sentiment_badge_html(sentimen: str, probabilitas: float | None = None) -> str`: kembalikan HTML `<span>` dengan `background-color` dari `get_sentiment_color`, `color:white`, `padding:4px 12px`, `border-radius:12px`, `font-weight:bold`; jika `probabilitas` tidak None, sertakan dalam format `(XX.X%)` di dalam badge
  - Implementasikan `validasi_input(teks: str) -> bool`: kembalikan `True` jika `teks.strip()` tidak kosong, `False` jika kosong atau hanya whitespace
  - _Requirements: 1.3, 5.4, 5.6_

- [x] 7. Implementasikan `page_overview(df_komentar, df_penceramah)`
  - Tampilkan `st.title("Sistem Analisis Sentimen Konten Dakwah Digital — Kota Bandung")` dan `st.markdown()` untuk deskripsi penelitian serta info peneliti (`NAMA_PENELITI`, `NIM`)
  - Buat diagram pipeline tiga tahap menggunakan `st.columns(3)` dengan ikon teks: "📥 Scraping", "🧹 Cleaning+EDA", "🤖 Labeling+Naive Bayes"; tambahkan tanda panah `→` di antara kolom; tidak ada gambar eksternal
  - Hitung metrik dinamis dari dataframe:
    ```python
    total_komentar    = len(df_komentar)
    jumlah_penceramah = df_komentar[KOLOM_CHANNEL].nunique()
    jumlah_topik      = df_komentar[KOLOM_TOPIK].nunique()
    df_test           = df_komentar[df_komentar[KOLOM_SPLIT] == 'test']
    akurasi           = (df_test[KOLOM_LABEL] == df_test[KOLOM_PREDIKSI]).mean()
    ```
  - Tampilkan keempat metrik dalam `st.columns(4)` menggunakan `st.metric()`
  - Tambahkan try-except di awal fungsi: tangkap `FileNotFoundError` dan `Exception` dari `load_data()`, tampilkan `st.error(f"❌ {e}")` dan panggil `st.stop()`
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 8. Implementasikan `page_eda(df_komentar, df_video)`
  - Tambahkan `st.sidebar.multiselect("Filter Penceramah", options=sorted(df_komentar[KOLOM_CHANNEL].unique()))` dan `st.sidebar.multiselect("Filter Topik", options=sorted(df_komentar[KOLOM_TOPIK].unique()))`; terapkan filter ke `df_filtered` sebelum setiap visualisasi dirender; jika tidak ada filter yang dipilih, gunakan seluruh data
  - Implementasikan enam visualisasi Plotly menggunakan `st.plotly_chart(fig, use_container_width=True)`:
    1. `px.bar` distribusi jumlah komentar per Sentimen (sumbu x: label sentimen, sumbu y: jumlah)
    2. `px.bar` horizontal Top 15–20 penceramah berdasarkan jumlah komentar
    3. `px.pie` distribusi topik dakwah
    4. `px.bar` distribusi topik dakwah
    5. `go.Figure(go.Heatmap(...))` penceramah × topik (pivot `channel_name` vs `topik_utama`, nilai: jumlah komentar)
    6. `px.bar` horizontal Top 30 kata paling sering dari `komentar_bersih` menggunakan `collections.Counter`
  - Implementasikan word cloud: gabungkan teks `df_filtered[KOLOM_KOMENTAR_BERSIH].dropna()`, buat `WordCloud(width=800, height=400, background_color='white').generate(teks_gabung)`, render ke `matplotlib` figure, tampilkan via `st.image()`
  - Implementasikan expander EDA Report PNG: cek `list(Path(DIR_EDA_REPORT).glob("*.png"))`; jika ada file PNG tampilkan `st.expander("Lihat chart asli dari notebook")` berisi `st.image()` per file; jika tidak ada, sembunyikan expander secara silent tanpa pesan error
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 9. Implementasikan `page_evaluasi(df_komentar)`
  - Tambahkan guard clause di awal: `df_test = df_komentar[df_komentar[KOLOM_SPLIT] == 'test']`; jika `df_test.empty` → `st.info("Data split test tidak tersedia di CSV_Komentar.")` dan `return`
  - Hitung confusion matrix menggunakan `sklearn.metrics.confusion_matrix(y_true, y_pred, labels=["Positif", "Negatif", "Netral"])` dan tampilkan sebagai `go.Figure(go.Heatmap(...))` Plotly dengan label sumbu x (Prediksi) dan sumbu y (Aktual)
  - Hitung empat metrik menggunakan fungsi scikit-learn dengan `average='weighted', zero_division=0`: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`; tampilkan dalam `st.columns(4)` sebagai `st.metric()`
  - Tampilkan tabel classification report: `pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).T` menggunakan `st.dataframe()`
  - Tampilkan tabel misklasiﬁkasi: `df_salah = df_test[df_test[KOLOM_LABEL] != df_test[KOLOM_PREDIKSI]]`; sorot seluruh baris yang salah prediksi dengan warna latar merah menggunakan Pandas Styler (`style.apply` atau `style.highlight_max`)
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 10. Implementasikan `page_demo(model_nb, vectorizer, stemmer, stopword_remover)`
  - Tambahkan `st.selectbox` dengan ≥5 contoh komentar khas dakwah yang representatif; ketika pengguna memilih contoh, isi `st.text_area` menggunakan `st.session_state` atau nilai default `value=` dari selectbox
  - Tampilkan `st.text_area("Masukkan komentar:", ...)` dan `st.button("Klasifikasi Sekarang")`
  - Pada klik tombol: panggil `validasi_input(teks)`; jika `False` → `st.warning("Input tidak boleh kosong.")` dan hentikan eksekusi (`return` atau `st.stop()`)
  - Jalankan pipeline klasifikasi langkah demi langkah dan simpan hasil antara:
    ```python
    teks_bersih  = cleaning_pipeline(teks_asli)
    teks_nostop  = stopword_remover.remove(teks_bersih)
    teks_stem    = stemmer.stem(teks_nostop)
    fitur_tfidf  = vectorizer.transform([teks_stem])
    label_pred   = model_nb.predict(fitur_tfidf)[0]
    proba        = model_nb.predict_proba(fitur_tfidf)[0]
    ```
  - Tampilkan badge: `st.markdown(get_sentiment_badge_html(label_pred, max(proba)), unsafe_allow_html=True)`
  - Tampilkan bar chart probabilitas tiga kelas menggunakan `px.bar` (sumbu x: nama kelas, sumbu y: probabilitas 0–1)
  - Tampilkan `st.expander("Lihat langkah preprocessing step-by-step")` dengan empat tahap: teks asli, setelah cleaning, setelah stopword removal, setelah stemming
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_

- [x] 11. Implementasikan `page_pencarian(df_komentar, df_video, df_penceramah)`
  - Tampilkan `st.selectbox` searchable dengan opsi `sorted(df_penceramah[KOLOM_CHANNEL].unique())`
  - Hitung dan tampilkan kartu profil penceramah yang dipilih dari `df_komentar[df_komentar[KOLOM_CHANNEL] == penceramah]`:
    - Total video dan total komentar
    - Donut chart distribusi Sentimen menggunakan `px.pie(hole=0.5)`
    - Badge sentimen dominan menggunakan `get_sentiment_badge_html()`
  - Filter `df_video[df_video[KOLOM_CHANNEL] == penceramah]`; untuk setiap video tampilkan: judul video, topik dakwah, tanggal publish, ringkasan konten, tautan `▶️ Tonton di YouTube` dengan URL `https://www.youtube.com/watch?v={video_id}`, dan mini `px.bar` distribusi sentimen komentar video tersebut (filter `df_komentar` by `video_id`)
  - Jika penceramah tidak memiliki video di CSV_Video → `st.info("Tidak ada data video untuk penceramah ini.")`
  - Implementasikan detail video ketika pengguna memilih satu video (via `st.selectbox` atau `st.expander`): `st.video(f"https://www.youtube.com/watch?v={video_id}")`, pie/bar chart sentimen, tabel sampel komentar per kelas sentimen, dan word cloud opsional (jika ada komentar)
  - Implementasikan fallback: jika `video_id` kosong atau NaN → filter CSV_Komentar berdasarkan `channel_name + topik_utama` dan tampilkan `st.info("Menggunakan filter channel + topik karena video_id tidak tersedia.")`
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

- [x] 12. Implementasikan error handling global dan finalkan routing utama
  - Pastikan setiap fungsi halaman menangkap `FileNotFoundError` dan `Exception` dari `load_data()` / `load_model()` dengan pola:
    ```python
    try:
        df_komentar, df_video, df_penceramah = load_data()
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memuat data: {e}")
        st.stop()
    ```
  - Pastikan setiap halaman hanya menerima argumen yang dibutuhkan (bukan semua data sekaligus)
  - Finalkan blok routing: baca `pilihan = st.sidebar.radio(...)`, dispatch ke fungsi halaman yang sesuai dengan argumen yang tepat
  - Verifikasi `streamlit run app.py` berjalan tanpa `NameError`, `ImportError`, atau `AttributeError` saat startup (dengan data nyata atau minimal)
  - _Requirements: 1.2, 1.6_

- [x] 13. Buat file `patch_cell12.py`
  - Buat file `patch_cell12.py` di root proyek berisi kode Python lengkap Cell 12 yang sudah dipatch
  - Definisikan `KOLOM_KOMENTAR` dengan `'video_id'` sebagai elemen pertama, diikuti kolom lainnya
  - Definisikan `KOLOM_VIDEO` dengan `'video_id'` sebagai elemen pertama, diikuti kolom lainnya
  - Ubah operasi groupby `sentimen_per_video` agar menggunakan `['video_id', 'channel_name', 'topik_utama', 'sentiment_label']` sebagai kunci
  - Ubah operasi merge back agar menggunakan `['video_id', 'channel_name', 'topik_utama', 'sentiment_label']` sebagai kunci join
  - Sertakan komentar `# SEBELUM PATCH` dan `# SETELAH PATCH` untuk setiap perubahan
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 14. Tulis unit tests di `tests/test_unit.py`
  - Test `load_data` dengan path tidak valid: assert `FileNotFoundError` dilempar dan pesan mengandung path absolut file yang tidak ditemukan
  - Test guard clause `page_evaluasi`: buat `pd.DataFrame` dengan kolom `split`, `sentiment_label`, `label_prediksi` hanya berisi baris `'train'`; assert `df_test.empty == True`
  - Test `validasi_input()`: `validasi_input("") == False`, `validasi_input("   ") == False`, `validasi_input("komentar valid") == True`
  - Test fallback `video_id`: assert `bool("")` adalah `False` dan `pd.isna(float('nan'))` adalah `True` — keduanya memicu fallback
  - Test `get_sentiment_badge_html` dengan `probabilitas=None`: assert HTML yang dikembalikan tidak mengandung karakter `%`
  - Test `get_sentiment_badge_html` dengan `probabilitas=0.85`: assert HTML yang dikembalikan mengandung `"85.0%"`
  - Test `cleaning_pipeline` dengan input konkret: `cleaning_pipeline("Hello World! 😀")` → hasil hanya berisi `[a-z ]` dan tidak ada leading/trailing whitespace
  - _Requirements: 1.6, 4.5, 5.6, 6.6_

- [x] 15. Tulis property-based tests di `tests/test_properties.py`

  - [x] 15.1 Property 1 — `cleaning_pipeline` output charset
    - Gunakan `@given(st.text())` dengan `@settings(max_examples=500)`
    - Assert `re.fullmatch(r'[a-z ]*', hasil) is not None` dan `hasil == hasil.strip()`
    - **Validates: Requirements 5.3, 5.8**

  - [x] 15.2 Property 2 — `cleaning_pipeline` idempoten
    - Gunakan `@given(st.text())` dengan `@settings(max_examples=500)`
    - Assert `cleaning_pipeline(cleaning_pipeline(t)) == cleaning_pipeline(t)`
    - **Validates: Requirements 5.3, 5.8**

  - [x] 15.3 Property 3 — `get_sentiment_color` konsisten untuk semua sentimen valid
    - Gunakan `@given(st.sampled_from(["Positif", "Negatif", "Netral"]))`
    - Assert hasil tidak kosong, diawali `"#"`, dan panjangnya tepat 7 karakter
    - **Validates: Requirements 1.3, 5.4**

  - [x] 15.4 Property 4 — kalkulasi akurasi konsisten dengan ground truth
    - Generate `pd.DataFrame` menggunakan `@given(st.lists(st.fixed_dictionaries({"split": st.sampled_from(["train", "test"]), "sentiment_label": st.sampled_from(["Positif", "Negatif", "Netral"]), "label_prediksi": st.sampled_from(["Positif", "Negatif", "Netral"])}), min_size=1))`
    - Assert akurasi yang dihitung `== (df_test[KOLOM_LABEL] == df_test[KOLOM_PREDIKSI]).mean()` dan `0.0 <= akurasi <= 1.0`
    - **Validates: Requirements 2.3, 2.4**

  - [x] 15.5 Property 5 — filter misklasiﬁkasi mengembalikan tepat baris salah
    - Generate DataFrame dengan skema yang sama seperti Property 4
    - Filter `df_salah = df_test[df_test[KOLOM_LABEL] != df_test[KOLOM_PREDIKSI]]`
    - Assert `len(df_salah) == jumlah baris di mana label != prediksi` dan tidak ada baris dengan `sentiment_label == label_prediksi` di `df_salah`
    - **Validates: Requirements 4.4**

  - [x] 15.6 Property 6 — daftar penceramah selalu terurut alfabetis
    - Gunakan `@given(st.lists(st.text(min_size=1), min_size=1))`
    - Assert `all(lst[i] <= lst[i+1] for i in range(len(lst)-1))` untuk daftar hasil `sorted(...)`
    - **Validates: Requirements 6.1**

  - [x] 15.7 Property 7 — format link YouTube selalu kanonik
    - Gunakan `@given(st.text(min_size=1, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='-_')))`
    - Assert link yang dihasilkan `== f"https://www.youtube.com/watch?v={video_id}"` — tidak ada spasi, encoding, atau karakter tambahan
    - **Validates: Requirements 6.3**

- [x] 16. Lengkapi dokumentasi komentar di `app.py` dan buat `README.md`
  - Pastikan komentar header `app.py` sudah mencakup: cara instalasi, cara menjalankan, tata letak direktori lengkap, dan cara mengedit `NAMA_PENELITI`/`NIM`
  - Buat `README.md` di root proyek dengan: deskripsi singkat aplikasi, prasyarat Python dan versi, langkah instalasi dan menjalankan, daftar lima halaman, dan catatan bahwa patch `patch_cell12.py` perlu diterapkan ke Cell 12 Script 3 Colab sebelum mengekspor CSV
  - _Requirements: 1.7, 8.2_

- [x] 18. Buat halaman `page_manajemen_penceramah()` — skeleton dan routing
  - Tambahkan opsi `"Manajemen Penceramah"` ke `st.sidebar.radio` di `main()` (setelah opsi `"Pencarian Penceramah & Sentimen Video"`)
  - Buat fungsi stub `page_manajemen_penceramah()` yang berisi `st.title("⚙️ Manajemen Penceramah")` dan `st.write("TODO")`
  - Tambahkan routing di `main()` untuk mendispatch ke `page_manajemen_penceramah()` ketika opsi tersebut dipilih
  - _Requirements: 1.2_

- [x] 19. Implementasikan fitur tambah penceramah
  - Tambahkan `st.text_input("Nama channel penceramah baru:")` dan `st.button("➕ Tambah Penceramah")` di dalam `page_manajemen_penceramah()`
  - Validasi input: nama tidak boleh kosong (`validasi_input()`) dan belum ada di kolom `channel_name` pada `data/dataset_final_per_penceramah.csv`; tampilkan `st.error()` jika validasi gagal
  - Jika valid: append baris baru ke `data/dataset_final_per_penceramah.csv` dengan semua kolom numerik (`total_komentar`, `jumlah_positif`, `jumlah_negatif`, `jumlah_netral`, `pct_positif`, `pct_negatif`, `pct_netral`) diset ke `0` dan `sentimen_dominan` diset ke `"N/A"`; gunakan `pd.read_csv` → concat → `to_csv(index=False)`
  - Panggil `st.cache_data.clear()` setelah menulis ke CSV, lalu panggil `st.rerun()` untuk memuat ulang data
  - Tampilkan `st.success(f"Penceramah '{nama}' berhasil ditambahkan.")` atau `st.error(f"❌ Gagal menambahkan penceramah: {e}")` sesuai hasil
  - _Requirements: 1.4, 1.6_

- [x] 20. Implementasikan fitur hapus penceramah
  - Tambahkan `st.selectbox("Pilih penceramah yang akan dihapus:", options=sorted(df_penceramah[KOLOM_CHANNEL].unique()))` di dalam `page_manajemen_penceramah()`
  - Tambahkan `st.checkbox("Saya yakin ingin menghapus penceramah ini beserta semua data terkait")` sebagai konfirmasi; tombol hapus hanya aktif (`disabled=not confirmed`) ketika checkbox dicentang
  - Pada penghapusan yang dikonfirmasi: hapus baris yang cocok dengan `channel_name` dari `data/dataset_final_per_penceramah.csv`; hapus semua baris yang cocok dari `data/dataset_final_komentar.csv` (alias `hasil_prediksi_lengkap.csv`) dan `data/dataset_final_video.csv`; gunakan pola `df[df[kolom] != nama].to_csv(path, index=False)` untuk setiap file
  - Panggil `st.cache_data.clear()` dan `st.rerun()` setelah semua file diperbarui
  - Tampilkan `st.success(f"Penceramah '{nama}' dan semua data terkait berhasil dihapus.")` atau `st.error(f"❌ Gagal menghapus penceramah: {e}")` sesuai hasil
  - _Requirements: 1.4, 1.6_

- [x] 21. Tulis unit tests untuk fitur manajemen penceramah di `tests/test_unit.py`
  - Test tambah penceramah dengan nama duplikat: buat DataFrame sementara dengan satu baris penceramah, lalu assert bahwa logika validasi mendeteksi nama yang sudah ada dan menolak penambahan
  - Test tambah penceramah dengan nama kosong: assert `validasi_input("")` mengembalikan `False` sehingga penambahan ditolak
  - Test hapus penceramah: buat CSV sementara menggunakan `tmp_path` pytest berisi beberapa baris, jalankan logika penghapusan untuk satu `channel_name`, baca kembali CSV hasil, dan assert bahwa baris tersebut tidak ada lagi sementara baris lain tetap ada
  - _Requirements: 1.6_

- [x] 17. Reorganisasi struktur folder proyek
  - Pindahkan `hasil_model/dataset_final_*.csv` dan `hasil_model/hasil_prediksi_lengkap.csv` dan `hasil_model/ringkasan_hasil.json` → `data/`
  - Pindahkan `hasil_model/model_naive_bayes.pkl` dan `hasil_model/tfidf_vectorizer.pkl` → `model/`
  - Pindahkan `hasil_model/grafik_*.png` → `eda_report/` (digabung dengan EDA chart)
  - Hapus folder `hasil_model/` yang sudah kosong
  - Pindahkan `raw_data_*.csv`, `clean_data_*.csv`, `progress_checkpoint*.json` → `data/raw/` (data mentah dan intermediate)
  - Pindahkan `Script*.ipynb` → `notebooks/` dengan nama file dibersihkan (spasi dan nomor versi dihapus)
  - Bersihkan `__pycache__/` dan `tests/__pycache__/`
  - _Requirements: 8.2, 8.3_

---

## Notes

- Property tests menggunakan library **Hypothesis** (`hypothesis==6.104.2`)
- Unit tests menggunakan **pytest** (`pytest==8.2.2`)
- Komponen UI Streamlit (`st.metric`, `st.plotly_chart`, `st.image`) tidak diuji langsung — fokus pada logika pure function yang tidak bergantung pada Streamlit runtime
- Pemuatan model `.pkl` diuji dengan kondisi file tidak ada (path fiktif), bukan dengan model produksi
- Implementasi monolitik di `app.py` dipilih untuk konsistensi `st.cache_resource` lintas halaman (lihat design doc)
- Gunakan path relatif untuk semua akses file data/model agar portabel (Requirements 8.3)

---

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2"] },
    { "id": 2, "tasks": ["3"] },
    { "id": 3, "tasks": ["4"] },
    { "id": 4, "tasks": ["5"] },
    { "id": 5, "tasks": ["6"] },
    { "id": 6, "tasks": ["7", "8", "9", "10", "11"] },
    { "id": 7, "tasks": ["12"] },
    { "id": 8, "tasks": ["13"] },
    { "id": 9, "tasks": ["14", "15.1", "15.2", "15.3", "15.4", "15.5", "15.6", "15.7"] },
    { "id": 10, "tasks": ["16"] },
    { "id": 11, "tasks": ["17"] },
    { "id": 12, "tasks": ["18"] },
    { "id": 13, "tasks": ["19", "20"] },
    { "id": 14, "tasks": ["21"] }
  ]
}
```
