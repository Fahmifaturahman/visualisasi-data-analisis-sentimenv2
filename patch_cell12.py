# ============================================================
# patch_cell12.py
# ============================================================
# Patch untuk Cell 12 di Script 3 Google Colab.
# Terapkan kode ini ke Cell 12 sebelum mengekspor CSV agar
# kolom video_id tersedia pada semua file output CSV.
#
# CARA PENGGUNAAN:
#   Salin kode di bagian "SETELAH PATCH" ke Cell 12 Script 3 Colab
#   sebagai pengganti baris-baris yang ditandai "SEBELUM PATCH".
# ============================================================

# ----------------------------------------------------------
# 1. Definisi KOLOM_KOMENTAR
# ----------------------------------------------------------

# SEBELUM PATCH:
# KOLOM_KOMENTAR = ['channel_name', 'topik_utama', 'komentar_bersih',
#                   'comment_text', 'sentiment_label', 'label_prediksi', 'split']

# SETELAH PATCH:
KOLOM_KOMENTAR = [
    'video_id',          # ← TAMBAH: pastikan video_id tersimpan di CSV_Komentar
    'channel_name',
    'topik_utama',
    'komentar_bersih',
    'comment_text',
    'sentiment_label',
    'label_prediksi',
    'split',
]

# ----------------------------------------------------------
# 2. Definisi KOLOM_VIDEO
# ----------------------------------------------------------

# SEBELUM PATCH:
# KOLOM_VIDEO = ['channel_name', 'topik_utama', 'judul_video',
#                'ringkasan_konten', 'tanggal_publish']

# SETELAH PATCH:
KOLOM_VIDEO = [
    'video_id',          # ← TAMBAH: pastikan video_id tersimpan di CSV_Video
    'channel_name',
    'topik_utama',
    'judul_video',
    'ringkasan_konten',
    'tanggal_publish',
]

# ----------------------------------------------------------
# 3. Operasi groupby sentimen_per_video
# ----------------------------------------------------------

# SEBELUM PATCH:
# sentimen_per_video = (
#     df_labeled
#     .groupby(['channel_name', 'topik_utama', 'sentiment_label'])
#     .size()
#     .reset_index(name='jumlah')
# )

# SETELAH PATCH:
sentimen_per_video = (
    df_labeled
    .groupby(['video_id', 'channel_name', 'topik_utama', 'sentiment_label'])  # ← UBAH: tambah video_id sebagai kunci
    .size()
    .reset_index(name='jumlah')
)

# ----------------------------------------------------------
# 4. Operasi merge back
# ----------------------------------------------------------

# SEBELUM PATCH:
# df_labeled = df_labeled.merge(
#     sentimen_per_video,
#     on=['channel_name', 'topik_utama', 'sentiment_label'],
#     how='left'
# )

# SETELAH PATCH:
df_labeled = df_labeled.merge(
    sentimen_per_video,
    on=['video_id', 'channel_name', 'topik_utama', 'sentiment_label'],  # ← UBAH: gunakan video_id sebagai kunci join
    how='left'
)
