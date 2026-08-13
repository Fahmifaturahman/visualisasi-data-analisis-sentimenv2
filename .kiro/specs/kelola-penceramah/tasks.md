# Implementation Plan: Kelola Penceramah

## Overview

Implementasi fitur **Kelola Penceramah** pada `app.py` dengan menambahkan:
1. Fungsi-fungsi helper murni (`validasi_input_penceramah`, `hitung_agregat_sentimen`, `tambah_penceramah_ke_csv`, `hapus_penceramah_dari_csv`)
2. Fungsi halaman `page_kelola_penceramah()` yang merender seluruh UI halaman
3. Integrasi ke `main()` melalui sidebar navigasi yang sudah ada
4. Unit tests dan property-based tests (Hypothesis) di direktori `tests/`

Semua operasi tulis menggunakan pola atomic write (`os.replace`) dan diikuti `st.cache_data.clear()`.

---

## Tasks

- [ ] 1. Implementasi fungsi helper murni (pure functions)
  - [ ] 1.1 Implementasi `validasi_input_penceramah`
    - Tulis fungsi di `app.py` dengan signature:
      `validasi_input_penceramah(channel_name, jumlah_positif, jumlah_negatif, jumlah_netral, df_existing) -> tuple[bool, str]`
    - Validasi berurutan: strip kosong → non-negatif → duplikat case-insensitive
    - Kembalikan `(True, "")` jika semua valid; `(False, pesan_error)` pada kesalahan pertama
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ]* 1.2 Tulis unit tests untuk `validasi_input_penceramah`
    - Kasus: channel_name kosong `""` dan `"   "` → `(False, "Nama penceramah tidak boleh kosong.")`
    - Kasus: channel_name valid, belum ada di df → `(True, "")`
    - Kasus: duplikat exact match → `(False, "Penceramah dengan nama tersebut sudah ada.")`
    - Kasus: duplikat case berbeda `"ustadz A"` vs `"Ustadz A"` → error duplikat
    - Kasus: duplikat whitespace extra → error duplikat
    - Tambahkan ke `tests/test_unit.py` (kelompok `TestValidasiInputPenceramah`)
    - _Requirements: 2.2, 2.4, 2.6_

  - [ ]* 1.3 Tulis property test untuk `validasi_input_penceramah` (Property 4 & 5)
    - **Property 4: Input whitespace-only selalu ditolak**
    - **Validates: Requirements 2.2, 2.4**
    - **Property 5: Deteksi duplikat bersifat case-insensitive dan strip-insensitive**
    - **Validates: Requirements 2.6**
    - Tambahkan ke `tests/test_properties.py` dengan `@settings(max_examples=100)`
    - _Requirements: 2.2, 2.4, 2.6_

  - [ ] 1.4 Implementasi `hitung_agregat_sentimen`
    - Tulis fungsi di `app.py` dengan signature:
      `hitung_agregat_sentimen(jumlah_positif, jumlah_negatif, jumlah_netral) -> dict`
    - Hitung `total_komentar`, `pct_*` (1 desimal, 0.0 jika total=0), `sentimen_dominan`
    - Tiebreak: jika dua atau lebih label seri, `sentimen_dominan = "Positif"`
    - _Requirements: 2.7, 2.8, 2.9_

  - [ ]* 1.5 Tulis unit tests untuk `hitung_agregat_sentimen`
    - Kasus: `(10, 5, 5)` → total=20, pct_positif=50.0, dominan="Positif"
    - Kasus: `(0, 0, 0)` → total=0, semua pct=0.0, dominan="Positif" (seri)
    - Kasus: `(3, 5, 2)` → dominan="Negatif"
    - Kasus: `(5, 5, 3)` → seri Positif & Negatif → dominan="Positif"
    - Kasus: `(0, 0, 7)` → dominan="Netral"
    - Kasus pembulatan: `(1, 0, 2)` → total=3, pct_positif=33.3, pct_netral=66.7
    - Tambahkan ke `tests/test_unit.py` (kelompok `TestHitungAgregatSentimen`)
    - _Requirements: 2.7, 2.8, 2.9_

  - [ ]* 1.6 Tulis property tests untuk `hitung_agregat_sentimen` (Property 1, 2, 3)
    - **Property 1: Total komentar adalah jumlah dari ketiga komponen**
    - **Validates: Requirements 2.7**
    - **Property 2: Persentase selalu berjumlah tepat 100 (atau semua nol jika total nol)**
    - **Validates: Requirements 2.8**
    - **Property 3: Sentimen dominan selalu merupakan label dengan jumlah tertinggi**
    - **Validates: Requirements 2.9**
    - Tambahkan ke `tests/test_properties.py` dengan `@settings(max_examples=100)`
    - _Requirements: 2.7, 2.8, 2.9_

- [ ] 2. Implementasi operasi I/O CSV (atomic write)
  - [ ] 2.1 Implementasi `tambah_penceramah_ke_csv`
    - Tulis fungsi di `app.py` dengan signature:
      `tambah_penceramah_ke_csv(path_csv: str, baris_baru: dict) -> None`
    - Muat CSV dari disk (bukan cache), append baris baru, simpan dengan atomic write:
      tulis ke `<path>.tmp` → `os.replace(<path>.tmp, path_csv)`
    - Lempar `IOError` jika operasi gagal; hapus file tmp jika ada
    - Gunakan `df.to_csv(path, index=True)` untuk mempertahankan format kolom index
    - _Requirements: 2.10, 2.12_

  - [ ] 2.2 Implementasi `hapus_penceramah_dari_csv`
    - Tulis fungsi di `app.py` dengan signature:
      `hapus_penceramah_dari_csv(path_csv: str, channel_name: str) -> None`
    - Muat CSV dari disk, hapus baris dengan case-sensitive match, simpan atomic write
    - Lempar `IOError` jika operasi gagal; file asli tidak berubah karena atomic write
    - _Requirements: 3.3, 3.6_

  - [ ]* 2.3 Tulis integration tests untuk `tambah_penceramah_ke_csv`
    - Gunakan `tmp_path` fixture pytest untuk file CSV sementara
    - Skenario: tambah ke CSV kosong → baris muncul di CSV
    - Skenario: tambah ke CSV berisi data → baris ditambahkan, baris lama tetap
    - Skenario: simulasi IOError (file read-only) → `IOError` dilempar, file tidak berubah
    - Tambahkan ke `tests/test_unit.py` (kelompok `TestTambahPenceramahKeCsv`)
    - _Requirements: 2.10, 2.12_

  - [ ]* 2.4 Tulis integration tests untuk `hapus_penceramah_dari_csv`
    - Skenario: hapus penceramah yang ada → baris hilang, baris lain tetap
    - Skenario: hapus dari CSV satu baris → CSV menjadi kosong (hanya header)
    - Skenario: simulasi IOError → `IOError` dilempar, file asli tidak berubah
    - Tambahkan ke `tests/test_unit.py` (kelompok `TestHapusPenceramahDariCsv`)
    - _Requirements: 3.3, 3.6_

- [ ] 3. Checkpoint — pastikan semua tests lulus
  - Jalankan `pytest tests/test_unit.py tests/test_properties.py -v` dan pastikan semua lulus.
  - Perbaiki kegagalan sebelum melanjutkan ke task berikutnya, tanyakan kepada user jika ada keraguan.

- [ ] 4. Implementasi fungsi halaman `page_kelola_penceramah`
  - [ ] 4.1 Render struktur halaman dan tabel data
    - Tulis fungsi `page_kelola_penceramah() -> None` di `app.py`
    - Muat `df_penceramah` dengan error handling: `st.error()` + return jika file tidak ditemukan
    - Tampilkan judul halaman dan ringkasan jumlah penceramah (Requirements 1.3)
    - Tampilkan tabel `df_penceramah` dengan kolom lengkap menggunakan `st.dataframe` (Requirements 4.1, 4.2)
    - _Requirements: 1.2, 1.3, 1.4, 4.1, 4.2, 4.4_

  - [ ] 4.2 Implementasi bagian Tambah Penceramah (Formulir_Tambah)
    - Tambahkan `st.subheader("Tambah Penceramah")` dan form input di bawah tabel
    - Gunakan `st.text_input` (maks 255 char) untuk `channel_name` dan `st.number_input` (0–999_999_999, step=1) untuk tiga kolom jumlah
    - Pada klik tombol simpan: panggil `validasi_input_penceramah`, tampilkan `st.error` jika gagal
    - Jika valid: panggil `hitung_agregat_sentimen` → `tambah_penceramah_ke_csv` → `st.cache_data.clear()` → pesan sukses → reset form
    - Tangkap `IOError` dari `tambah_penceramah_ke_csv` dan tampilkan `st.error` (Requirements 2.12)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 5.1, 5.3_

  - [ ] 4.3 Implementasi bagian Hapus Penceramah
    - Tambahkan `st.subheader("Hapus Penceramah")` dan peringatan informasional tetap (Requirements 3.7)
    - Tampilkan `st.selectbox` daftar penceramah terurut alfabetis; nonaktifkan jika df kosong (Requirements 3.1, 3.8)
    - Tampilkan `st.checkbox` konfirmasi dan tombol hapus; nonaktifkan tombol jika df kosong (Requirements 3.2, 3.4, 3.8)
    - Pada klik tombol hapus: cek checkbox → jika tidak dicentang `st.warning`; jika dicentang: `hapus_penceramah_dari_csv` → `st.cache_data.clear()` → pesan sukses
    - Tangkap `IOError` dari `hapus_penceramah_dari_csv` dan tampilkan `st.error` (Requirements 3.6)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 5.2, 5.3_

- [ ] 5. Integrasi ke `main()` dan verifikasi akhir
  - [ ] 5.1 Tambah routing "Kelola Penceramah" ke `main()`
    - Tambahkan `"Kelola Penceramah"` ke list `options` pada `st.sidebar.radio`
    - Tambahkan `elif pilihan == "Kelola Penceramah": page_kelola_penceramah()` di blok routing
    - _Requirements: 1.1, 1.2_

  - [ ]* 5.2 Tulis regression tests untuk routing sidebar
    - Verifikasi `"Kelola Penceramah"` hadir di options sidebar (mock `st.sidebar.radio`)
    - Tambahkan ke `tests/test_unit.py` (kelompok `TestSidebarRouting`)
    - _Requirements: 1.1_

- [ ] 6. Checkpoint akhir — pastikan semua tests lulus
  - Jalankan `pytest tests/ -v` dan pastikan seluruh test suite lulus.
  - Perbaiki kegagalan sebelum menyatakan implementasi selesai, tanyakan kepada user jika ada keraguan.

---

## Notes

- Tasks bertanda `*` bersifat opsional dan dapat dilewati untuk iterasi cepat
- Setiap task mereferensikan requirements spesifik untuk keterlacakan
- Property tests menggunakan Hypothesis dengan `@settings(max_examples=100)`
- Atomic write (`os.replace`) memastikan tidak ada korupsi file CSV saat crash
- `st.cache_data.clear()` dipanggil **setelah** write sukses, **sebelum** pesan sukses
- Semua error ditampilkan inline (`st.error`/`st.warning`) — tidak ada `st.stop()` pada error form
- Test file I/O menggunakan `tmp_path` fixture pytest, tidak menyentuh data produksi

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.4"] },
    { "id": 1, "tasks": ["1.2", "1.3", "1.5", "1.6", "2.1", "2.2"] },
    { "id": 2, "tasks": ["2.3", "2.4", "4.1"] },
    { "id": 3, "tasks": ["4.2", "4.3"] },
    { "id": 4, "tasks": ["5.1"] },
    { "id": 5, "tasks": ["5.2"] }
  ]
}
```
