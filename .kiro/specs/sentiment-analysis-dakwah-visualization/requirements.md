# Dokumen Kebutuhan

## Pendahuluan

Sistem Analisis Sentimen Konten Dakwah Digital — Kota Bandung adalah aplikasi web berbasis Streamlit yang menampilkan seluruh alur sistem analisis sentimen komentar YouTube dari konten dakwah Islam penceramah di Bandung. Aplikasi ini mengintegrasikan hasil pipeline yang telah dibangun di Google Colab (scraping, cleaning+EDA, labeling+Naive Bayes) ke dalam antarmuka interaktif dengan lima halaman utama: Overview, EDA Dashboard, Evaluasi Model, Demo Klasifikasi Real-Time, dan Pencarian Penceramah & Sentimen Video.

## Glosarium

- **Aplikasi**: Aplikasi web Streamlit "Sistem Analisis Sentimen Konten Dakwah Digital — Kota Bandung"
- **Pengguna**: Pengunjung yang mengakses antarmuka web Aplikasi
- **CSV_Komentar**: File `data/dataset_final_komentar.csv` yang berisi data komentar berlabel dengan kolom `video_id`, `channel_name`, `topik_utama`, `komentar_bersih`, `sentiment_label`, `label_prediksi`, `split`
- **CSV_Video**: File `data/dataset_final_video.csv` yang berisi metadata video dengan kolom `video_id`, `channel_name`, `topik_utama`, `judul_video`, `ringkasan_konten`, `tanggal_publish`
- **CSV_Penceramah**: File `data/dataset_final_per_penceramah.csv` yang berisi data agregat per penceramah dengan kolom `channel_name`, `total_video`, `total_komentar`, `sentimen_dominan`
- **Model_NB**: File `model/model_naive_bayes.pkl` yang berisi model Multinomial Naive Bayes yang telah dilatih
- **Vectorizer**: File `model/tfidf_vectorizer.pkl` yang berisi TF-IDF vectorizer yang telah dilatih
- **EDA_Report**: Direktori `eda_report/` yang berisi 9 file PNG chart hasil notebook Google Colab
- **Preprocessing_Pipeline**: Fungsi `cleaning_pipeline` dan `full_preprocessing` yang menerapkan langkah pembersihan teks, penghapusan stopword, dan stemming Sastrawi identik dengan Script 2 dan Script 3 Colab
- **Sentimen**: Label klasifikasi teks, bernilai salah satu dari: Positif, Negatif, atau Netral
- **Penceramah**: Tokoh dakwah Islam di Bandung yang videonya dikumpulkan, diidentifikasi lewat `channel_name`
- **Split_Test**: Baris pada CSV_Komentar di mana nilai kolom `split` adalah `'test'`
- **Sidebar**: Panel navigasi kiri Streamlit yang menampilkan pilihan halaman dan filter global

---

## Kebutuhan

### Kebutuhan 1: Konfigurasi dan Inisialisasi Aplikasi

**User Story:** Sebagai Pengguna, saya ingin aplikasi dapat dimuat dengan konfigurasi yang benar, sehingga saya mendapatkan tampilan yang konsisten dan responsif.

#### Kriteria Penerimaan

1. THE Aplikasi SHALL mengatur konfigurasi halaman dengan `layout="wide"`, judul tab browser "Sistem Analisis Sentimen Konten Dakwah Digital — Kota Bandung", dan ikon halaman yang sesuai saat pertama kali dimuat.
2. THE Aplikasi SHALL menampilkan navigasi sidebar berisi lima pilihan halaman: "Overview", "EDA Dashboard", "Evaluasi Model", "Demo Klasifikasi Real-Time", dan "Pencarian Penceramah & Sentimen Video".
3. THE Aplikasi SHALL menggunakan palet warna konsisten: hijau untuk Sentimen Positif, merah untuk Sentimen Negatif, dan abu-abu untuk Sentimen Netral di seluruh halaman.
4. THE Aplikasi SHALL memuat CSV_Komentar, CSV_Video, dan CSV_Penceramah menggunakan `st.cache_data` agar tidak membaca ulang file dari disk pada setiap interaksi.
5. THE Aplikasi SHALL memuat Model_NB dan Vectorizer menggunakan `st.cache_resource` agar objek tidak di-reinisialisasi pada setiap interaksi.
6. IF file CSV_Komentar, CSV_Video, CSV_Penceramah, Model_NB, atau Vectorizer tidak ditemukan pada path yang diharapkan, THEN THE Aplikasi SHALL menampilkan pesan error yang menyertakan path lengkap file yang dicari dan menghentikan eksekusi halaman terkait.
7. THE Aplikasi SHALL menyertakan komentar instruksi di bagian atas `app.py` yang menjelaskan cara menjalankan aplikasi dan tata letak direktori yang diperlukan.

---

### Kebutuhan 2: Halaman Overview

**User Story:** Sebagai Pengguna, saya ingin melihat ringkasan penelitian dan statistik utama dataset, sehingga saya dapat memahami konteks dan skala data yang dianalisis.

#### Kriteria Penerimaan

1. THE Aplikasi SHALL menampilkan judul "Sistem Analisis Sentimen Konten Dakwah Digital — Kota Bandung" dan paragraf deskripsi singkat penelitian pada Halaman Overview.
2. THE Aplikasi SHALL menampilkan diagram alur pipeline tiga tahap — Scraping, Cleaning+EDA, dan Labeling+Naive Bayes — menggunakan kolom Streamlit horizontal dengan ikon teks dan tanda panah, tanpa memuat gambar eksternal.
3. WHEN Halaman Overview ditampilkan, THE Aplikasi SHALL menghitung dan menampilkan empat kartu `st.metric()` secara dinamis dari CSV_Komentar dan CSV_Penceramah: total komentar, jumlah penceramah unik, jumlah topik dakwah unik, dan akurasi model pada Split_Test.
4. THE Aplikasi SHALL menghitung akurasi model pada kartu metrik dengan membandingkan kolom `sentiment_label` dan `label_prediksi` hanya pada baris di mana `split == 'test'` dalam CSV_Komentar, bukan menggunakan nilai yang di-hardcode.

---

### Kebutuhan 3: Halaman EDA Dashboard

**User Story:** Sebagai Pengguna, saya ingin menjelajahi visualisasi interaktif dari data komentar, sehingga saya dapat memahami karakteristik dataset secara mendalam.

#### Kriteria Penerimaan

1. THE Aplikasi SHALL menampilkan enam jenis visualisasi Plotly pada Halaman EDA Dashboard yang dirender dari CSV_Komentar dan CSV_Video: distribusi jumlah komentar per Sentimen, Top 15–20 penceramah berdasarkan jumlah komentar, distribusi topik dakwah (pie chart dan bar chart), heatmap penceramah × topik, dan grafik Top 30 kata paling sering muncul pada kolom `komentar_bersih`.
2. THE Aplikasi SHALL menampilkan word cloud menggunakan library `wordcloud` yang dirender dari teks pada kolom `komentar_bersih` di CSV_Komentar dan ditampilkan via `st.image`.
3. THE Aplikasi SHALL menyediakan filter multiselect di Sidebar untuk `channel_name` dan `topik_utama`; WHEN filter diubah, THE Aplikasi SHALL memperbarui semua visualisasi pada Halaman EDA Dashboard agar mencerminkan subset data yang dipilih.
4. WHEN direktori EDA_Report tersedia dan file PNG yang relevan ditemukan, THE Aplikasi SHALL menampilkan expander berlabel "Lihat chart asli dari notebook" yang menampilkan gambar PNG dari EDA_Report.
5. IF direktori EDA_Report atau file PNG individual tidak ditemukan, THEN THE Aplikasi SHALL menyembunyikan expander tanpa menampilkan pesan error untuk elemen opsional tersebut.

---

### Kebutuhan 4: Halaman Evaluasi Model

**User Story:** Sebagai Pengguna, saya ingin melihat performa model Multinomial Naive Bayes secara mendetail, sehingga saya dapat menilai kualitas dan reliabilitas klasifikasi sentimen.

#### Kriteria Penerimaan

1. WHEN Halaman Evaluasi Model ditampilkan, THE Aplikasi SHALL menghitung confusion matrix dari baris Split_Test pada CSV_Komentar dan menampilkannya sebagai Plotly heatmap dengan label kelas Positif, Negatif, dan Netral.
2. WHEN Halaman Evaluasi Model ditampilkan, THE Aplikasi SHALL menghitung dan menampilkan empat kartu metrik — Accuracy, Precision (weighted avg), Recall (weighted avg), dan F1-Score (weighted avg) — dari baris Split_Test pada CSV_Komentar menggunakan fungsi evaluasi scikit-learn.
3. THE Aplikasi SHALL menampilkan tabel classification report per kelas (Positif, Negatif, Netral) dengan kolom precision, recall, f1-score, dan support, dihitung dari baris Split_Test pada CSV_Komentar.
4. THE Aplikasi SHALL menampilkan tabel "Prediksi Benar vs Salah" yang memuat sampel baris dari Split_Test, di mana baris dengan `sentiment_label` tidak sama dengan `label_prediksi` disorot dengan warna latar merah menggunakan Pandas Styler.
5. IF tidak ada baris dengan `split == 'test'` pada CSV_Komentar, THEN THE Aplikasi SHALL menampilkan pesan informatif bahwa data split test tidak tersedia dan menghentikan rendering halaman.

---

### Kebutuhan 5: Halaman Demo Klasifikasi Real-Time

**User Story:** Sebagai Pengguna, saya ingin mengetik atau menempel komentar bebas dan melihat hasil klasifikasi sentimennya secara langsung, sehingga saya dapat memverifikasi cara kerja model pada teks baru.

#### Kriteria Penerimaan

1. THE Aplikasi SHALL menampilkan text area untuk input teks bebas dan tombol berlabel "Klasifikasi Sekarang" pada Halaman Demo Klasifikasi Real-Time.
2. THE Aplikasi SHALL menyediakan sejumlah contoh komentar siap pakai yang dapat dipilih Pengguna melalui selectbox atau tombol, sehingga teks contoh tersebut mengisi text area secara otomatis.
3. WHEN Pengguna mengklik tombol "Klasifikasi Sekarang" dengan text area tidak kosong, THE Aplikasi SHALL menjalankan Preprocessing_Pipeline secara berurutan: `cleaning_pipeline` (hapus URL, mention, emoji, karakter non-alfabet, case folding, normalisasi spasi) kemudian `full_preprocessing` (cleaning → penghapusan stopword Sastrawi → stemming Sastrawi).
4. WHEN Pengguna mengklik tombol "Klasifikasi Sekarang" dengan text area tidak kosong, THE Aplikasi SHALL menampilkan label Sentimen hasil klasifikasi sebagai badge berwarna (hijau untuk Positif, merah untuk Negatif, abu-abu untuk Netral) dan bar chart Plotly yang menampilkan probabilitas ketiga kelas Sentimen.
5. WHEN Pengguna mengklik tombol "Klasifikasi Sekarang" dengan text area tidak kosong, THE Aplikasi SHALL menampilkan expander "Lihat langkah preprocessing step-by-step" yang memperlihatkan teks pada setiap tahap pipeline: teks asli, setelah `cleaning_pipeline`, setelah penghapusan stopword, dan setelah stemming.
6. IF Pengguna mengklik tombol "Klasifikasi Sekarang" dengan text area kosong, THEN THE Aplikasi SHALL menampilkan pesan peringatan bahwa input tidak boleh kosong dan tidak menjalankan klasifikasi.
7. THE Aplikasi SHALL menginisialisasi stemmer Sastrawi (`StemmerFactory`) dan stopword remover Sastrawi (`StopWordRemoverFactory`) menggunakan `st.cache_resource` agar tidak di-reinisialisasi pada setiap interaksi.
8. THE Preprocessing_Pipeline yang dijalankan pada Halaman Demo Klasifikasi Real-Time SHALL identik secara logika dengan pipeline yang digunakan pada Script 2 dan Script 3 Google Colab.

---

### Kebutuhan 6: Halaman Pencarian Penceramah & Sentimen Video

**User Story:** Sebagai Pengguna, saya ingin mencari penceramah tertentu dan menelusuri video serta analisis sentimen komentarnya, sehingga saya dapat memahami persepsi audiens terhadap konten dakwah penceramah tersebut.

#### Kriteria Penerimaan

1. THE Aplikasi SHALL menampilkan search box atau selectbox yang dapat dicari (searchable) berisi nama-nama penceramah unik dari kolom `channel_name` pada CSV_Penceramah, yang telah diurutkan secara alfabetis.
2. WHEN Pengguna memilih nama penceramah, THE Aplikasi SHALL menampilkan kartu profil penceramah yang memuat: total video, total komentar, donut chart distribusi Sentimen keseluruhan, dan badge Sentimen dominan — semua nilai dihitung dari CSV_Komentar yang difilter berdasarkan `channel_name` yang dipilih.
3. WHEN Pengguna memilih nama penceramah, THE Aplikasi SHALL menampilkan daftar video penceramah tersebut dari CSV_Video yang difilter berdasarkan `channel_name`; setiap entri video SHALL menampilkan judul video, topik dakwah, tanggal publish, ringkasan konten (`ringkasan_konten`), tautan "▶️ Tonton di YouTube" menggunakan URL `https://www.youtube.com/watch?v={video_id}`, dan mini chart Plotly distribusi Sentimen komentar video tersebut (dihitung dari CSV_Komentar yang difilter `video_id`).
4. WHEN Pengguna memilih video tertentu dari daftar, THE Aplikasi SHALL menampilkan tampilan detail video yang memuat: embed video melalui `st.video(f"https://www.youtube.com/watch?v={video_id}")`, pie atau bar chart distribusi Sentimen komentar video tersebut, dan tabel sampel komentar yang dikelompokkan per kelas Sentimen.
5. WHERE data komentar untuk video yang dipilih memiliki lebih dari nol baris pada CSV_Komentar, THE Aplikasi SHALL menampilkan word cloud opsional dari komentar video tersebut.
6. IF `video_id` untuk entri video tidak tersedia atau kosong, THEN THE Aplikasi SHALL memfilter CSV_Komentar berdasarkan `channel_name` dan `topik_utama` sebagai fallback dan menampilkan banner informasi yang menjelaskan bahwa filter alternatif sedang digunakan.
7. IF Pengguna memilih nama penceramah yang tidak memiliki data video pada CSV_Video, THEN THE Aplikasi SHALL menampilkan pesan informatif bahwa tidak ada data video untuk penceramah tersebut.

---

### Kebutuhan 7: Patch Data Pipeline (Script 3 Cell 12)

**User Story:** Sebagai peneliti, saya ingin data output dari pipeline Colab menyertakan `video_id` pada semua file CSV, sehingga fitur pencarian berdasarkan video dapat berfungsi dengan benar di Aplikasi.

#### Kriteria Penerimaan

1. THE Patch SHALL menambahkan string `'video_id'` ke dalam list `KOLOM_KOMENTAR` pada Cell 12 Script 3 Colab sehingga kolom `video_id` ikut tersimpan pada CSV_Komentar.
2. THE Patch SHALL menambahkan string `'video_id'` ke dalam list `KOLOM_VIDEO` pada Cell 12 Script 3 Colab sehingga kolom `video_id` ikut tersimpan pada CSV_Video.
3. THE Patch SHALL mengubah operasi groupby pada variabel `sentimen_per_video` agar menggunakan `['video_id', 'channel_name', 'topik_utama', 'sentiment_label']` sebagai kunci pengelompokan, menggantikan `['channel_name', 'topik_utama', 'sentiment_label']`.
4. THE Patch SHALL mengubah operasi merge kembali (merge back) dari `sentimen_per_video` ke dataframe utama agar menggunakan `video_id` sebagai kunci join, bukan kombinasi `channel_name` + `topik_utama`.

---

### Kebutuhan 8: Paket dan Distribusi

**User Story:** Sebagai peneliti, saya ingin file dependensi yang lengkap dan instruksi yang jelas, sehingga Aplikasi dapat dijalankan di lingkungan baru tanpa kesulitan konfigurasi.

#### Kriteria Penerimaan

1. THE Aplikasi SHALL menyertakan file `requirements.txt` yang mencantumkan seluruh dependensi Python dengan versi yang di-pin secara eksplisit: `streamlit`, `pandas`, `numpy`, `plotly`, `Pillow`, `PySastrawi`, `scikit-learn`, dan `wordcloud`.
2. THE Aplikasi SHALL dapat dijalankan dengan perintah tunggal `streamlit run app.py` dari direktori root proyek tanpa konfigurasi tambahan selain ketersediaan file data dan model pada path yang ditentukan.
3. THE Aplikasi SHALL menggunakan path relatif terhadap direktori root proyek untuk mengakses semua file data (`data/`), file model (`model/`), dan file EDA (`eda_report/`) agar portabel di berbagai sistem operasi.
