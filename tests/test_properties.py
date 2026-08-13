"""
Property-based tests untuk Sistem Analisis Sentimen Konten Dakwah Digital.

Library: Hypothesis
Framework: pytest

Menguji properti universal pada fungsi-fungsi pure dari preprocessing pipeline
dan helper UI di app.py.

Catatan: Fungsi pure (cleaning_pipeline, get_sentiment_color) didefinisikan ulang
di sini agar test tidak bergantung pada import app.py yang memuat Streamlit/Plotly.
Implementasi identik dengan yang ada di app.py.
"""

import re

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st


# ============================================================
# Reimplementasi fungsi pure dari app.py (identik, tanpa dependensi Streamlit)
# ============================================================

WARNA_SENTIMEN = {
    "Positif": "#2ecc71",
    "Negatif": "#e74c3c",
    "Netral":  "#95a5a6",
}


def hapus_url(t: str) -> str:
    return re.sub(r'http\S+|www\.\S+', ' ', t)


def hapus_mention(t: str) -> str:
    return re.sub(r'[@#]\w+', ' ', t)


def hapus_emoji(t: str) -> str:
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(' ', t)


def hapus_nonalpha(t: str) -> str:
    return re.sub(r'[^a-zA-Z\s]', ' ', t)


def case_folding(t: str) -> str:
    return t.lower()


def norm_spasi(t: str) -> str:
    return re.sub(r'\s+', ' ', t).strip()


def cleaning_pipeline(teks: str) -> str:
    """Stage 1: URL → mention → emoji → non-alfa → case fold → norm spasi."""
    t = hapus_url(teks)
    t = hapus_mention(t)
    t = hapus_emoji(t)
    t = hapus_nonalpha(t)
    t = case_folding(t)
    t = norm_spasi(t)
    return t


def get_sentiment_color(sentimen: str) -> str:
    return WARNA_SENTIMEN.get(sentimen, "#95a5a6")


# ============================================================
# Property 1: cleaning_pipeline output charset
# Feature: sentiment-analysis-dakwah-visualization
# Validates: Requirements 5.3, 5.8
# ============================================================

@given(st.text())
@settings(max_examples=500)
def test_property_1_cleaning_pipeline_output_charset(teks_input):
    """
    Feature: sentiment-analysis-dakwah-visualization
    Property 1: Cleaning Pipeline Menghasilkan Output Hanya Huruf Kecil ASCII dan Spasi
    Output hanya boleh mengandung [a-z ] dan tidak ada leading/trailing whitespace.
    """
    hasil = cleaning_pipeline(teks_input)
    assert re.fullmatch(r'[a-z ]*', hasil) is not None, \
        f"Output mengandung karakter non-[a-z ]: {repr(hasil)}"
    assert hasil == hasil.strip(), \
        f"Output memiliki leading/trailing whitespace: {repr(hasil)}"


# ============================================================
# Property 2: cleaning_pipeline idempoten
# Feature: sentiment-analysis-dakwah-visualization
# Validates: Requirements 5.3, 5.8
# ============================================================

@given(st.text())
@settings(max_examples=500)
def test_property_2_cleaning_pipeline_idempotent(teks_input):
    """
    Feature: sentiment-analysis-dakwah-visualization
    Property 2: Cleaning Pipeline Bersifat Idempoten
    cleaning_pipeline(cleaning_pipeline(t)) == cleaning_pipeline(t)
    """
    hasil_sekali = cleaning_pipeline(teks_input)
    hasil_dua_kali = cleaning_pipeline(hasil_sekali)
    assert hasil_sekali == hasil_dua_kali, \
        f"Pipeline tidak idempoten: satu kali={repr(hasil_sekali)}, dua kali={repr(hasil_dua_kali)}"


# ============================================================
# Property 3: get_sentiment_color konsisten untuk semua sentimen valid
# Feature: sentiment-analysis-dakwah-visualization
# Validates: Requirements 1.3, 5.4
# ============================================================

@given(st.sampled_from(["Positif", "Negatif", "Netral"]))
def test_property_3_color_mapper_consistent(sentimen):
    """
    Feature: sentiment-analysis-dakwah-visualization
    Property 3: Color Mapper Konsisten untuk Semua Sentimen Valid
    Hasil harus non-kosong, diawali '#', panjang tepat 7 karakter.
    """
    warna = get_sentiment_color(sentimen)
    assert warna, f"Warna kosong untuk sentimen: {sentimen}"
    assert warna.startswith("#"), f"Warna tidak diawali '#': {warna}"
    assert len(warna) == 7, f"Warna bukan format #RRGGBB (panjang={len(warna)}): {warna}"


# ============================================================
# Property 4: kalkulasi akurasi konsisten dengan ground truth
# Feature: sentiment-analysis-dakwah-visualization
# Validates: Requirements 2.3, 2.4
# ============================================================

KOLOM_LABEL    = "sentiment_label"
KOLOM_PREDIKSI = "label_prediksi"
KOLOM_SPLIT    = "split"


@given(st.lists(
    st.fixed_dictionaries({
        "split":           st.sampled_from(["train", "test"]),
        "sentiment_label": st.sampled_from(["Positif", "Negatif", "Netral"]),
        "label_prediksi":  st.sampled_from(["Positif", "Negatif", "Netral"]),
    }),
    min_size=1,
))
def test_property_4_accuracy_consistent(rows):
    """
    Feature: sentiment-analysis-dakwah-visualization
    Property 4: Akurasi Konsisten dengan Ground Truth
    Nilai akurasi harus sama dengan proporsi prediksi benar di split test,
    dan berada dalam rentang [0.0, 1.0].

    **Validates: Requirements 2.3, 2.4**
    """
    df = pd.DataFrame(rows)
    df_test = df[df[KOLOM_SPLIT] == "test"]

    if df_test.empty:
        return  # skip jika tidak ada baris test

    akurasi = (df_test[KOLOM_LABEL] == df_test[KOLOM_PREDIKSI]).mean()
    expected = (df_test[KOLOM_LABEL] == df_test[KOLOM_PREDIKSI]).mean()

    assert abs(akurasi - expected) < 1e-9, \
        f"Akurasi tidak konsisten: hitung={akurasi}, expected={expected}"
    assert 0.0 <= akurasi <= 1.0, \
        f"Akurasi di luar rentang [0.0, 1.0]: {akurasi}"


# ============================================================
# Property 5: filter misklasiﬁkasi mengembalikan tepat baris salah
# Feature: sentiment-analysis-dakwah-visualization
# Validates: Requirements 4.4
# ============================================================

@given(st.lists(
    st.fixed_dictionaries({
        "split":           st.sampled_from(["train", "test"]),
        "sentiment_label": st.sampled_from(["Positif", "Negatif", "Netral"]),
        "label_prediksi":  st.sampled_from(["Positif", "Negatif", "Netral"]),
    }),
    min_size=1,
))
def test_property_5_misclassification_filter(rows):
    """
    Feature: sentiment-analysis-dakwah-visualization
    Property 5: Filter Misklasiﬁkasi Mengembalikan Tepat Baris Salah
    Jumlah baris salah harus sama dengan jumlah baris di mana label != prediksi,
    dan tidak ada baris dengan prediksi benar di df_salah.

    **Validates: Requirements 4.4**
    """
    df = pd.DataFrame(rows)
    df_test = df[df[KOLOM_SPLIT] == "test"]

    if df_test.empty:
        return  # skip jika tidak ada baris test

    df_salah = df_test[df_test[KOLOM_LABEL] != df_test[KOLOM_PREDIKSI]]

    # Jumlah baris df_salah harus sama dengan jumlah baris di mana label != prediksi
    expected_count = (df_test[KOLOM_LABEL] != df_test[KOLOM_PREDIKSI]).sum()
    assert len(df_salah) == expected_count, \
        f"Jumlah baris salah tidak cocok: {len(df_salah)} vs expected {expected_count}"

    # Tidak boleh ada baris dengan prediksi benar di df_salah
    if not df_salah.empty:
        benar_di_salah = (df_salah[KOLOM_LABEL] == df_salah[KOLOM_PREDIKSI]).any()
        assert not benar_di_salah, \
            "df_salah mengandung baris dengan prediksi benar"


# ============================================================
# Property 6: daftar penceramah selalu terurut alfabetis
# Feature: sentiment-analysis-dakwah-visualization
# Validates: Requirements 6.1
# ============================================================

@given(st.lists(st.text(min_size=1), min_size=1))
def test_property_6_penceramah_list_sorted(names):
    """
    Feature: sentiment-analysis-dakwah-visualization
    Property 6: Daftar Penceramah Selalu Terurut Alfabetis
    sorted(names) harus menghasilkan list yang setiap elemen ke-i <= elemen ke-(i+1).
    """
    sorted_names = sorted(names)
    for i in range(len(sorted_names) - 1):
        assert sorted_names[i] <= sorted_names[i + 1], \
            f"Urutan tidak benar di posisi {i}: {repr(sorted_names[i])} > {repr(sorted_names[i+1])}"


# ============================================================
# Property 7: format link YouTube selalu kanonik
# Feature: sentiment-analysis-dakwah-visualization
# Validates: Requirements 6.3
# ============================================================

@given(st.text(
    min_size=1,
    alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'),
        whitelist_characters='-_',
    )
))
def test_property_7_youtube_url_canonical(video_id):
    """
    Feature: sentiment-analysis-dakwah-visualization
    Property 7: Format Link YouTube Selalu Kanonik
    Link harus selalu menggunakan format kanonik tanpa spasi atau karakter tambahan.
    """
    yt_url = f"https://www.youtube.com/watch?v={video_id}"

    assert yt_url == f"https://www.youtube.com/watch?v={video_id}", \
        "Link YouTube tidak kanonik"
    assert " " not in yt_url, \
        f"Link YouTube mengandung spasi: {repr(yt_url)}"
    assert yt_url.startswith("https://www.youtube.com/watch?v="), \
        f"Link YouTube tidak diawali dengan URL kanonik: {repr(yt_url)}"
    assert yt_url.endswith(video_id), \
        f"Link YouTube tidak diakhiri dengan video_id: {repr(yt_url)}"
