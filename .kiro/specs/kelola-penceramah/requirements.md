# Requirements Document

## Introduction

Fitur **Kelola Penceramah** menambahkan kemampuan manajemen data penceramah secara langsung dari dalam dashboard Streamlit. Saat ini, data penceramah hanya bisa diubah dengan mengedit file CSV secara manual di luar aplikasi. Fitur ini memungkinkan pengguna untuk menambahkan entri penceramah baru ke dalam `dataset_final_per_penceramah.csv` serta menghapus entri penceramah yang sudah ada — tanpa perlu keluar dari aplikasi atau memanipulasi CSV secara manual.

Konteks sistem:
- Aplikasi ini adalah dashboard Streamlit untuk analisis sentimen komentar YouTube pada konten dakwah Islam di Kota Bandung.
- Data penceramah agregat tersimpan di `data/dataset_final_per_penceramah.csv` (kolom: `channel_name`, `total_komentar`, `jumlah_positif`, `jumlah_negatif`, `jumlah_netral`, `pct_positif`, `pct_negatif`, `pct_netral`, `sentimen_dominan`).
- Data komentar tersimpan di `data/hasil_prediksi_lengkap.csv` dengan kolom `channel_name` sebagai referensi penceramah.
- Data video tersimpan di `data/dataset_final_video.csv` dengan kolom `channel_name` sebagai referensi penceramah.
- Streamlit menggunakan `@st.cache_data` untuk caching data; perubahan pada CSV memerlukan clear cache agar terefleksi di UI.

---

## Glossary

- **Penceramah**: Entitas yang mewakili satu channel YouTube milik seorang pendakwah Islam, diidentifikasi secara unik dengan `channel_name`.
- **Dashboard**: Aplikasi Streamlit (`app.py`) yang menjadi antarmuka utama pengguna.
- **CSV_Penceramah**: File `data/dataset_final_per_penceramah.csv` yang menyimpan data agregat sentimen per penceramah.
- **CSV_Komentar**: File `data/hasil_prediksi_lengkap.csv` yang menyimpan seluruh komentar beserta label sentimen dan prediksi.
- **CSV_Video**: File `data/dataset_final_video.csv` yang menyimpan metadata video per penceramah.
- **Halaman_Kelola_Penceramah**: Halaman baru di Dashboard yang menampung seluruh fungsionalitas pengelolaan data penceramah.
- **Formulir_Tambah**: Komponen UI di Halaman_Kelola_Penceramah untuk memasukkan data penceramah baru.
- **Konfirmasi_Hapus**: Dialog atau mekanisme konfirmasi eksplisit sebelum operasi penghapusan dieksekusi.
- **Cache_Data**: Mekanisme caching Streamlit (`st.cache_data`) yang menyimpan data CSV di memori.

---

## Requirements

### Requirement 1: Navigasi ke Halaman Kelola Penceramah

**User Story:** Sebagai peneliti, saya ingin mengakses halaman pengelolaan penceramah dari sidebar navigasi, agar saya dapat mengelola data penceramah tanpa harus berpindah ke luar aplikasi.

#### Acceptance Criteria

1. THE Dashboard SHALL menampilkan opsi navigasi "Kelola Penceramah" pada sidebar radio navigasi yang sudah ada.
2. WHEN pengguna memilih "Kelola Penceramah" dari sidebar, THE Dashboard SHALL menampilkan Halaman_Kelola_Penceramah.
3. WHEN Halaman_Kelola_Penceramah ditampilkan, THE Dashboard SHALL menampilkan judul halaman, ringkasan jumlah penceramah yang berstatus aktif (tidak dihapus dari sistem), serta dua bagian terpisah: bagian tambah penceramah dan bagian hapus penceramah.
4. IF Halaman_Kelola_Penceramah gagal dimuat karena kesalahan sistem, THEN THE Dashboard SHALL menampilkan pesan error yang menjelaskan kegagalan tersebut tanpa mempengaruhi tampilan sidebar navigasi.

---

### Requirement 2: Menambahkan Penceramah Baru

**User Story:** Sebagai peneliti, saya ingin menambahkan entri penceramah baru ke dalam sistem, agar data penceramah yang belum tercatat dapat dimasukkan ke dalam dashboard.

#### Acceptance Criteria

1. THE Halaman_Kelola_Penceramah SHALL menampilkan Formulir_Tambah yang berisi kolom input: nama penceramah (`channel_name`) dengan batas maksimal 255 karakter, jumlah komentar positif, jumlah komentar negatif, dan jumlah komentar netral, masing-masing dengan rentang nilai bilangan bulat 0 hingga 999.999.999.
2. WHEN pengguna menekan tombol simpan pada Formulir_Tambah, THE Dashboard SHALL memvalidasi bahwa kolom `channel_name` tidak kosong setelah dilakukan strip whitespace.
3. WHEN pengguna menekan tombol simpan pada Formulir_Tambah, THE Dashboard SHALL memvalidasi bahwa nilai jumlah komentar positif, negatif, dan netral masing-masing adalah bilangan bulat non-negatif.
4. IF kolom `channel_name` kosong setelah validasi, THEN THE Dashboard SHALL menampilkan pesan error "Nama penceramah tidak boleh kosong." tanpa menyimpan data.
5. IF nilai jumlah komentar tidak valid (bukan bilangan bulat non-negatif), THEN THE Dashboard SHALL menampilkan pesan error yang menyebutkan kolom mana yang tidak valid tanpa menyimpan data.
6. IF `channel_name` yang dimasukkan sudah terdapat di CSV_Penceramah (perbandingan case-insensitive setelah strip), THEN THE Dashboard SHALL menampilkan pesan error "Penceramah dengan nama tersebut sudah ada." tanpa menyimpan data.
7. IF semua input valid dan `channel_name` belum ada, THEN THE Dashboard SHALL menghitung `total_komentar` sebagai jumlah dari `jumlah_positif + jumlah_negatif + jumlah_netral`.
8. IF semua input valid dan `channel_name` belum ada, THEN THE Dashboard SHALL menghitung persentase (`pct_positif`, `pct_negatif`, `pct_netral`) sebagai pembagian masing-masing jumlah terhadap `total_komentar` dikalikan 100, dibulatkan hingga satu desimal; IF `total_komentar` adalah 0, THEN THE Dashboard SHALL menetapkan seluruh nilai persentase sebagai 0.0.
9. IF semua input valid dan `channel_name` belum ada, THEN THE Dashboard SHALL menentukan `sentimen_dominan` sebagai label dengan jumlah terbanyak di antara Positif, Negatif, dan Netral; IF lebih dari satu label memiliki jumlah yang sama dan tertinggi, THEN THE Dashboard SHALL menetapkan `sentimen_dominan` sebagai "Positif".
10. IF semua input valid dan `channel_name` belum ada, THEN THE Dashboard SHALL menambahkan baris baru ke CSV_Penceramah dan menyimpan file CSV ke disk.
11. WHEN penyimpanan ke disk berhasil, THE Dashboard SHALL menghapus Cache_Data, menampilkan data terbaru di tabel penceramah, mereset kolom input Formulir_Tambah ke nilai awal, dan menampilkan pesan sukses yang menyebutkan nama penceramah yang baru ditambahkan.
12. IF terjadi kesalahan I/O saat menyimpan CSV (misalnya file tidak dapat ditulisi), THEN THE Dashboard SHALL menampilkan pesan error yang menjelaskan kegagalan tanpa mengubah status data di memori.

---

### Requirement 3: Menghapus Penceramah dari Sistem

**User Story:** Sebagai peneliti, saya ingin menghapus entri penceramah yang tidak relevan dari sistem, agar data pada dashboard tetap akurat dan tidak memuat penceramah yang sudah tidak diperlukan.

#### Acceptance Criteria

1. THE Halaman_Kelola_Penceramah SHALL menampilkan daftar seluruh penceramah yang ada di CSV_Penceramah dalam komponen selectbox, diurutkan secara alfabetis berdasarkan nama penceramah.
2. WHEN pengguna memilih penceramah dari selectbox dan menekan tombol hapus, THE Dashboard SHALL menampilkan Konfirmasi_Hapus berupa checkbox berteks "Saya yakin ingin menghapus penceramah ini." yang harus dicentang sebelum konfirmasi dapat dilanjutkan.
3. WHEN pengguna telah mencentang Konfirmasi_Hapus dan menekan tombol konfirmasi hapus, THE Dashboard SHALL menghapus baris yang memiliki `channel_name` sesuai pilihan dari CSV_Penceramah dan menyimpan file CSV ke disk.
4. IF pengguna menekan tombol konfirmasi hapus tanpa mencentang Konfirmasi_Hapus, THEN THE Dashboard SHALL menampilkan pesan peringatan yang menginformasikan bahwa konfirmasi harus dicentang terlebih dahulu, tanpa melakukan penghapusan.
5. WHEN penghapusan dari CSV_Penceramah berhasil disimpan ke disk, THE Dashboard SHALL menghapus seluruh Cache_Data yang berkaitan dengan daftar penceramah, memperbarui isi selectbox agar penceramah yang dihapus tidak lagi muncul, dan menampilkan pesan sukses yang menyebutkan nama penceramah yang dihapus.
6. IF terjadi kesalahan I/O saat menyimpan CSV setelah penghapusan, THEN THE Dashboard SHALL membatalkan operasi penulisan sehingga file CSV_Penceramah di disk tetap identik dengan kondisi sebelum operasi dimulai, dan menampilkan pesan error yang menjelaskan kegagalan tersebut.
7. THE Halaman_Kelola_Penceramah SHALL selalu menampilkan peringatan informasional yang menjelaskan bahwa penghapusan penceramah dari CSV_Penceramah tidak secara otomatis menghapus komentar dan video terkait dari CSV_Komentar dan CSV_Video.
8. IF CSV_Penceramah tidak memuat data penceramah, THEN THE Dashboard SHALL menonaktifkan komponen selectbox dan tombol hapus, serta menampilkan pesan informasional yang menyatakan bahwa tidak ada penceramah yang dapat dihapus.

---

### Requirement 4: Tampilan Tabel Data Penceramah Saat Ini

**User Story:** Sebagai peneliti, saya ingin melihat daftar lengkap penceramah yang tersimpan saat ini, agar saya dapat memverifikasi data sebelum dan sesudah melakukan operasi tambah atau hapus.

#### Acceptance Criteria

1. THE Halaman_Kelola_Penceramah SHALL menampilkan tabel berisi seluruh data dari CSV_Penceramah tanpa paginasi, dengan kolom: `channel_name`, `total_komentar`, `jumlah_positif`, `jumlah_negatif`, `jumlah_netral`, `pct_positif`, `pct_negatif`, `pct_netral`, dan `sentimen_dominan`.
2. IF CSV_Penceramah dapat dibaca tetapi tidak memiliki baris data, THEN THE Halaman_Kelola_Penceramah SHALL menampilkan header kolom tabel disertai pesan informasional bahwa belum ada data penceramah yang tersimpan.
3. WHEN data berhasil ditambahkan atau dihapus, THE Halaman_Kelola_Penceramah SHALL memperbarui tabel untuk mencerminkan kondisi terbaru dari CSV_Penceramah.
4. IF CSV_Penceramah tidak dapat dibaca (misalnya file tidak ditemukan), THEN THE Dashboard SHALL menampilkan pesan error yang menyebutkan path file yang bermasalah dan tidak merender konten tabel maupun elemen halaman lainnya.

---

### Requirement 5: Konsistensi Cache Setelah Operasi Tulis

**User Story:** Sebagai peneliti, saya ingin perubahan data penceramah langsung terlihat di seluruh halaman dashboard setelah operasi tambah atau hapus, agar saya tidak melihat data yang sudah tidak akurat di halaman lain.

#### Acceptance Criteria

1. WHEN operasi tambah penceramah berhasil disimpan ke disk, THE Dashboard SHALL memanggil `st.cache_data.clear()` sebelum menampilkan pesan sukses.
2. WHEN operasi hapus penceramah berhasil disimpan ke disk, THE Dashboard SHALL memanggil `st.cache_data.clear()` sebelum menampilkan pesan sukses.
3. WHEN `st.cache_data.clear()` dipanggil, THE Dashboard SHALL menampilkan notifikasi informatif (bukan error) kepada pengguna bahwa cache telah di-refresh dan halaman lain akan memuat data terbaru saat di-reload atau saat pengguna berpindah ke halaman tersebut berikutnya.
