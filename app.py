"""
===========================================================================
Sistem Analisis Sentimen Konten Dakwah Digital — Kota Bandung
===========================================================================

CARA INSTALASI:
    pip install -r requirements.txt

CARA MENJALANKAN:
    streamlit run app.py

TATA LETAK DIREKTORI YANG DIPERLUKAN:
    project_root/
    ├── app.py                              ← file ini
    ├── requirements.txt
    ├── data/
    │   ├── hasil_prediksi_lengkap.csv      ← data komentar + prediksi
    │   ├── dataset_final_video.csv         ← metadata video
    │   ├── dataset_final_per_penceramah.csv← data agregat per penceramah
    │   └── raw/
    │       └── raw_data_video.csv          ← data video mentah (untuk video_id)
    ├── model/
    │   ├── model_naive_bayes.pkl           ← model Naive Bayes terlatih
    │   └── tfidf_vectorizer.pkl            ← TF-IDF vectorizer terlatih
    └── eda_report/                         ← (opsional) chart PNG dari Colab

CARA MENGEDIT DATA PENELITI:
    Ubah nilai NAMA_PENELITI dan NIM di bagian konstanta di bawah ini.
===========================================================================
"""

# --- IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import io
import re
import collections
from pathlib import Path
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# ---------------------------------------------------------------------------
# SECTION 1: Konfigurasi Halaman (HARUS jadi perintah Streamlit pertama)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sistem Analisis Sentimen Konten Dakwah Digital — Kota Bandung",
    page_icon="🕌",
    layout="wide",
)

# ---------------------------------------------------------------------------
# SECTION 1B: Konstanta Global
# ---------------------------------------------------------------------------

# EDIT SESUAI DATA PENELITI
NAMA_PENELITI = "Fahmi Faturahman"
NIM           = "1222008"

# Palet warna sentimen
WARNA_SENTIMEN = {
    "Positif": "#2ecc71",   # hijau
    "Negatif": "#e74c3c",   # merah
    "Netral":  "#95a5a6",   # abu-abu
}

# Path data (relatif terhadap direktori root proyek)
PATH_CSV_KOMENTAR   = "data/hasil_prediksi_lengkap.csv"
PATH_CSV_VIDEO      = "data/dataset_final_video.csv"
PATH_CSV_PENCERAMAH = "data/dataset_final_per_penceramah.csv"
PATH_MODEL_NB       = "model/model_naive_bayes.pkl"
PATH_VECTORIZER     = "model/tfidf_vectorizer.pkl"
DIR_EDA_REPORT      = "eda_report"

# Nama kolom kanonik (sesuai CSV asli)
KOLOM_LABEL           = "sentiment_label"
KOLOM_PREDIKSI        = "label_prediksi"
KOLOM_SPLIT           = "split"
KOLOM_CHANNEL         = "channel_name"
KOLOM_TOPIK           = "topik_utama"
KOLOM_VIDEO_ID        = "video_id"
KOLOM_KOMENTAR_BERSIH = "comment_clean"

# ---------------------------------------------------------------------------
# SECTION 2: Fungsi Cache — load_data(), load_model(), init_sastrawi()
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> tuple:
    """
    Memuat tiga CSV dari disk. Mengembalikan (df_komentar, df_video, df_penceramah).
    Memperkaya df_video dengan video_id dari raw_data_video.csv.
    Melempar FileNotFoundError dengan path absolut jika file tidak ditemukan.
    """
    for path in [PATH_CSV_KOMENTAR, PATH_CSV_VIDEO, PATH_CSV_PENCERAMAH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File tidak ditemukan: {os.path.abspath(path)}")

    df_komentar   = pd.read_csv(PATH_CSV_KOMENTAR)
    df_video      = pd.read_csv(PATH_CSV_VIDEO)
    df_penceramah = pd.read_csv(PATH_CSV_PENCERAMAH)

    # Enrich df_video dengan video_id dari raw_data_video.csv
    # df_video sudah punya 'title' dan 'published_at' — hanya perlu tambah video_id
    # Merge via tiga kunci (channel_name + topik_utama + title) agar tidak ada duplikat
    _raw_path = "data/raw/raw_data_video.csv"
    if os.path.exists(_raw_path):
        _df_raw = pd.read_csv(
            _raw_path,
            usecols=["video_id", "title", "channel_name", "topik_utama"],
        )
        _df_raw_dedup = (
            _df_raw
            .sort_values("video_id")
            .drop_duplicates(subset=["channel_name", "topik_utama", "title"])
        )
        df_video = df_video.merge(
            _df_raw_dedup[["video_id", "channel_name", "topik_utama", "title"]],
            on=["channel_name", "topik_utama", "title"],
            how="left",
        )

    return df_komentar, df_video, df_penceramah


@st.cache_resource
def load_model() -> tuple:
    """
    Memuat model_naive_bayes.pkl dan tfidf_vectorizer.pkl.
    Mengembalikan (model_nb, vectorizer).
    """
    for path in [PATH_MODEL_NB, PATH_VECTORIZER]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File tidak ditemukan: {os.path.abspath(path)}")
    with open(PATH_MODEL_NB, "rb") as f:
        model_nb = pickle.load(f)
    with open(PATH_VECTORIZER, "rb") as f:
        vectorizer = pickle.load(f)
    return model_nb, vectorizer


@st.cache_resource
def init_sastrawi() -> tuple:
    """
    Menginisialisasi Sastrawi Stemmer dan StopWordRemover.
    Mengembalikan (stemmer, stopword_remover).
    """
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    except ImportError as e:
        raise ImportError(
            "PySastrawi tidak terinstall. Jalankan: pip install PySastrawi"
        ) from e
    stemmer          = StemmerFactory().create_stemmer()
    stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
    return stemmer, stopword_remover


# ---------------------------------------------------------------------------
# SECTION 3: Preprocessing Pipeline
# ---------------------------------------------------------------------------

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
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(' ', t)


def hapus_nonalpha(t: str) -> str:
    return re.sub(r'[^a-zA-Z\s]', ' ', t)


def case_folding(t: str) -> str:
    return t.lower()


def norm_spasi(t: str) -> str:
    return re.sub(r'\s+', ' ', t).strip()


def cleaning_pipeline(teks: str) -> str:
    """Stage 1: URL → mention → emoji → non-alfa → case fold → norm spasi.
    Output invariant: hanya karakter [a-z ] dan tanpa leading/trailing whitespace.
    """
    t = hapus_url(teks)
    t = hapus_mention(t)
    t = hapus_emoji(t)
    t = hapus_nonalpha(t)
    t = case_folding(t)
    t = norm_spasi(t)
    return t


def full_preprocessing(teks: str, stopword_remover, stemmer) -> str:
    """Stage 1-3: cleaning → stopword removal → stemming.
    Identik secara logika dengan pipeline Script 2 dan Script 3 Colab.
    """
    t = cleaning_pipeline(teks)
    t = stopword_remover.remove(t)
    t = stemmer.stem(t)
    return t


# ---------------------------------------------------------------------------
# SECTION 4: Helper UI
# ---------------------------------------------------------------------------

def get_sentiment_color(sentimen: str) -> str:
    """Mengembalikan kode hex warna dari WARNA_SENTIMEN. Default abu-abu."""
    return WARNA_SENTIMEN.get(sentimen, "#95a5a6")


def get_sentiment_badge_html(sentimen: str, probabilitas: float = None) -> str:
    """
    Mengembalikan HTML badge berwarna untuk label sentimen.
    Opsional: tampilkan probabilitas dalam badge.
    """
    warna     = get_sentiment_color(sentimen)
    teks_prob = f" ({probabilitas:.1%})" if probabilitas is not None else ""
    return (
        f'<span style="background-color:{warna};color:white;'
        f'padding:4px 12px;border-radius:12px;font-weight:bold;">'
        f'{sentimen}{teks_prob}</span>'
    )


def validasi_input(teks: str) -> bool:
    """Kembalikan True jika input tidak kosong setelah strip."""
    return bool(teks.strip())


def _render_wordcloud(teks_gabung: str, judul: str = "") -> None:
    """Render word cloud dari teks gabung dan tampilkan via st.image."""
    if not teks_gabung.strip():
        st.info("Tidak cukup teks untuk membuat word cloud.")
        return
    wc  = WordCloud(width=800, height=400, background_color="white").generate(teks_gabung)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    if judul:
        ax.set_title(judul, fontsize=14)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    st.image(buf, use_container_width=True)


# ---------------------------------------------------------------------------
# SECTION 5: Halaman
# ---------------------------------------------------------------------------

# ------------------------------------------------------------------
# 5.1  page_overview
# ------------------------------------------------------------------
def page_overview() -> None:
    try:
        df_komentar, _df_video, _df_penceramah = load_data()
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memuat data: {e}")
        st.stop()

    st.title("🕌 Sistem Analisis Sentimen Konten Dakwah Digital — Kota Bandung")
    st.markdown(
        f"""
        Aplikasi ini menyajikan hasil **analisis sentimen komentar YouTube** dari konten dakwah Islam
        penceramah di Kota Bandung. Pipeline analisis dibangun menggunakan tiga notebook Google Colab:
        scraping data, cleaning & EDA, serta pelabelan sentimen dan klasifikasi dengan **Multinomial Naive Bayes**.

        **Peneliti:** {NAMA_PENELITI} &nbsp;|&nbsp; **NIM:** {NIM}
        """
    )

    st.markdown("---")
    st.subheader("Pipeline Analisis")

    col1, arrow1, col2, arrow2, col3 = st.columns([3, 0.5, 3, 0.5, 3])
    with col1:
        st.markdown(
            """
            <div style="background:#f0f4ff;padding:20px;border-radius:10px;text-align:center">
            <h3>📥 Script 1</h3>
            <b>Scraping</b><br>
            Pengumpulan komentar YouTube via YouTube Data API v3 untuk penceramah dakwah Bandung
            </div>
            """,
            unsafe_allow_html=True,
        )
    with arrow1:
        st.markdown("<div style='text-align:center;font-size:2rem;padding-top:40px'>→</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(
            """
            <div style="background:#f0fff4;padding:20px;border-radius:10px;text-align:center">
            <h3>🧹 Script 2</h3>
            <b>Cleaning + EDA</b><br>
            Pembersihan teks, penghapusan stopword, stemming Sastrawi, dan eksplorasi data
            </div>
            """,
            unsafe_allow_html=True,
        )
    with arrow2:
        st.markdown("<div style='text-align:center;font-size:2rem;padding-top:40px'>→</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(
            """
            <div style="background:#fff8f0;padding:20px;border-radius:10px;text-align:center">
            <h3>🤖 Script 3</h3>
            <b>Labeling + Naive Bayes</b><br>
            Pelabelan sentimen semi-otomatis, pelatihan model MNB dengan TF-IDF, evaluasi performa
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Statistik Dataset")

    total_komentar    = len(df_komentar)
    jumlah_penceramah = df_komentar[KOLOM_CHANNEL].nunique()
    jumlah_topik      = df_komentar[KOLOM_TOPIK].nunique()
    df_test           = df_komentar[df_komentar[KOLOM_SPLIT] == "test"]
    if not df_test.empty:
        akurasi = (df_test[KOLOM_LABEL] == df_test[KOLOM_PREDIKSI]).mean()
        akurasi_str = f"{akurasi:.2%}"
    else:
        akurasi_str = "N/A"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("💬 Total Komentar",     f"{total_komentar:,}")
    m2.metric("🎙️ Jumlah Penceramah", str(jumlah_penceramah))
    m3.metric("📚 Topik Dakwah",       str(jumlah_topik))
    m4.metric("🎯 Akurasi Model",      akurasi_str)

    st.markdown("---")
    st.subheader("Distribusi Sentimen Keseluruhan")
    dist = df_komentar[KOLOM_LABEL].value_counts().reset_index()
    dist.columns = ["Sentimen", "Jumlah"]
    fig = px.bar(
        dist,
        x="Sentimen",
        y="Jumlah",
        color="Sentimen",
        color_discrete_map=WARNA_SENTIMEN,
        text="Jumlah",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Jumlah Komentar")
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------
# 5.2  page_eda
# ------------------------------------------------------------------
def page_eda() -> None:
    try:
        df_komentar, df_video, _df_penceramah = load_data()
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memuat data: {e}")
        st.stop()

    st.title("📊 EDA Dashboard")
    st.markdown("Eksplorasi data interaktif dengan visualisasi dari CSV_Komentar dan CSV_Video.")

    # --- Sidebar filter ---
    st.sidebar.markdown("### Filter EDA")
    filter_channel = st.sidebar.multiselect(
        "Filter Penceramah",
        options=sorted(df_komentar[KOLOM_CHANNEL].dropna().unique()),
    )
    filter_topik = st.sidebar.multiselect(
        "Filter Topik",
        options=sorted(df_komentar[KOLOM_TOPIK].dropna().unique()),
    )

    df_filtered = df_komentar.copy()
    if filter_channel:
        df_filtered = df_filtered[df_filtered[KOLOM_CHANNEL].isin(filter_channel)]
    if filter_topik:
        df_filtered = df_filtered[df_filtered[KOLOM_TOPIK].isin(filter_topik)]

    if df_filtered.empty:
        st.warning("⚠️ Tidak ada data yang sesuai dengan filter yang dipilih.")
        return

    st.caption(f"Menampilkan {len(df_filtered):,} komentar dari {len(df_komentar):,} total.")

    # --- Vis 1: Distribusi sentimen ---
    st.subheader("1. Distribusi Sentimen")
    dist_sent = df_filtered[KOLOM_LABEL].value_counts().reset_index()
    dist_sent.columns = ["Sentimen", "Jumlah"]
    fig1 = px.bar(
        dist_sent, x="Sentimen", y="Jumlah",
        color="Sentimen", color_discrete_map=WARNA_SENTIMEN,
        text="Jumlah",
    )
    fig1.update_traces(textposition="outside")
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    # --- Vis 2: Top penceramah ---
    st.subheader("2. Top Penceramah berdasarkan Jumlah Komentar")
    top_n = 20
    top_channel = (
        df_filtered[KOLOM_CHANNEL]
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    top_channel.columns = ["Penceramah", "Jumlah"]
    fig2 = px.bar(
        top_channel, x="Jumlah", y="Penceramah",
        orientation="h", text="Jumlah",
        color="Jumlah", color_continuous_scale="Blues",
    )
    fig2.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    # --- Vis 3 & 4: Distribusi topik ---
    st.subheader("3. Distribusi Topik Dakwah")
    col_pie, col_bar = st.columns(2)
    dist_topik = df_filtered[KOLOM_TOPIK].value_counts().reset_index()
    dist_topik.columns = ["Topik", "Jumlah"]

    with col_pie:
        fig3 = px.pie(dist_topik, names="Topik", values="Jumlah", title="Pie Chart")
        st.plotly_chart(fig3, use_container_width=True)
    with col_bar:
        fig4 = px.bar(
            dist_topik, x="Jumlah", y="Topik",
            orientation="h", title="Bar Chart", text="Jumlah",
        )
        fig4.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig4, use_container_width=True)

    # --- Vis 5: Heatmap penceramah x topik ---
    st.subheader("4. Heatmap Penceramah × Topik")
    pivot_data = (
        df_filtered.groupby([KOLOM_CHANNEL, KOLOM_TOPIK])
        .size()
        .reset_index(name="jumlah")
    )
    if not pivot_data.empty:
        pivot_matrix = pivot_data.pivot(
            index=KOLOM_CHANNEL, columns=KOLOM_TOPIK, values="jumlah"
        ).fillna(0)
        fig5 = go.Figure(
            go.Heatmap(
                z=pivot_matrix.values,
                x=pivot_matrix.columns.tolist(),
                y=pivot_matrix.index.tolist(),
                colorscale="Blues",
                hoverongaps=False,
            )
        )
        fig5.update_layout(
            xaxis_title="Topik",
            yaxis_title="Penceramah",
            height=max(400, len(pivot_matrix) * 20),
        )
        st.plotly_chart(fig5, use_container_width=True)

    # --- Vis 6: Top 30 kata ---
    st.subheader("5. Top 30 Kata Paling Sering (Komentar Bersih)")
    teks_semua = " ".join(df_filtered[KOLOM_KOMENTAR_BERSIH].dropna().astype(str))
    kata_list  = teks_semua.split()
    counter    = collections.Counter(kata_list)
    top30      = counter.most_common(30)
    if top30:
        df_top30 = pd.DataFrame(top30, columns=["Kata", "Frekuensi"])
        fig6 = px.bar(
            df_top30, x="Frekuensi", y="Kata",
            orientation="h", text="Frekuensi",
            color="Frekuensi", color_continuous_scale="Greens",
        )
        fig6.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig6, use_container_width=True)

    # --- Word Cloud ---
    st.subheader("6. Word Cloud")
    teks_wc = " ".join(df_filtered[KOLOM_KOMENTAR_BERSIH].dropna().astype(str))
    _render_wordcloud(teks_wc)

    # --- Expander EDA Report PNG ---
    png_files = sorted(Path(DIR_EDA_REPORT).glob("*.png")) if Path(DIR_EDA_REPORT).exists() else []
    if png_files:
        with st.expander("📁 Lihat chart asli dari notebook"):
            for png in png_files:
                st.image(str(png), caption=png.name, use_container_width=True)


# ------------------------------------------------------------------
# 5.3  page_evaluasi
# ------------------------------------------------------------------
def page_evaluasi() -> None:
    try:
        df_komentar, _df_video, _df_penceramah = load_data()
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memuat data: {e}")
        st.stop()

    st.title("📈 Evaluasi Model")

    df_test = df_komentar[df_komentar[KOLOM_SPLIT] == "test"]
    if df_test.empty:
        st.info("ℹ️ Data split test tidak tersedia di CSV_Komentar.")
        return

    y_true = df_test[KOLOM_LABEL]
    y_pred = df_test[KOLOM_PREDIKSI]
    kelas  = ["Positif", "Negatif", "Netral"]

    # --- Confusion Matrix ---
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred, labels=kelas)
    fig_cm = go.Figure(
        go.Heatmap(
            z=cm,
            x=[f"Prediksi: {k}" for k in kelas],
            y=[f"Aktual: {k}" for k in kelas],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            hoverongaps=False,
        )
    )
    fig_cm.update_layout(
        xaxis_title="Prediksi",
        yaxis_title="Aktual",
        height=400,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # --- Metrik ---
    st.subheader("Metrik Evaluasi")
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Accuracy",         f"{accuracy:.4f}")
    m2.metric("🔍 Precision (w.avg)", f"{precision:.4f}")
    m3.metric("📡 Recall (w.avg)",    f"{recall:.4f}")
    m4.metric("⚖️ F1-Score (w.avg)",  f"{f1:.4f}")

    # --- Classification Report ---
    st.subheader("Classification Report per Kelas")
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df_report   = pd.DataFrame(report_dict).T.round(4)
    st.dataframe(df_report, use_container_width=True)

    # --- Tabel Misklasiﬁkasi ---
    st.subheader("Tabel Misklasiﬁkasi (Prediksi Salah)")
    df_salah = df_test[df_test[KOLOM_LABEL] != df_test[KOLOM_PREDIKSI]].copy()
    if df_salah.empty:
        st.success("✅ Tidak ada prediksi yang salah pada data test!")
    else:
        kolom_tampil = [
            c for c in [
                "comment_text", KOLOM_CHANNEL, KOLOM_TOPIK,
                KOLOM_LABEL, KOLOM_PREDIKSI
            ] if c in df_salah.columns
        ]
        df_salah_tampil = df_salah[kolom_tampil].head(200)

        def sorot_salah(row):
            return ["background-color: #fde8e8"] * len(row)

        st.dataframe(
            df_salah_tampil.style.apply(sorot_salah, axis=1),
            use_container_width=True,
        )
        st.caption(
            f"Menampilkan {len(df_salah_tampil)} dari {len(df_salah):,} prediksi salah."
        )


# ------------------------------------------------------------------
# 5.4  page_demo
# ------------------------------------------------------------------
def page_demo() -> None:
    try:
        model_nb, vectorizer = load_model()
        stemmer, stopword_remover = init_sastrawi()
    except (FileNotFoundError, ImportError) as e:
        st.error(f"❌ {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memuat model: {e}")
        st.stop()

    st.title("🤖 Demo Klasifikasi Real-Time")
    st.markdown(
        "Masukkan komentar berbahasa Indonesia tentang konten dakwah dan klik "
        "**Klasifikasi Sekarang** untuk melihat prediksi sentimen secara langsung."
    )

    # Contoh komentar
    CONTOH_KOMENTAR = [
        "— Pilih contoh komentar —",
        "Masya Allah ceramahnya sangat menyentuh hati, semoga Allah merahmati ustadz",
        "Penjelasannya mudah dipahami dan sangat bermanfaat untuk kehidupan sehari-hari",
        "Suaranya tidak jelas dan materi yang disampaikan membingungkan",
        "Biasa saja, tidak ada yang istimewa dari ceramah ini",
        "Alhamdulillah ilmu yang sangat berharga, terima kasih sudah berbagi",
        "Ceramah ini tidak sesuai dengan ajaran yang benar menurut saya",
        "Konten yang sangat informatif dan inspiratif, terus berkarya ustadz",
    ]

    pilihan = st.selectbox("Contoh komentar siap pakai:", CONTOH_KOMENTAR)

    teks_default = "" if pilihan == CONTOH_KOMENTAR[0] else pilihan
    teks_asli = st.text_area(
        "Masukkan komentar:",
        value=teks_default,
        height=120,
        placeholder="Ketik atau tempel komentar di sini...",
    )

    if st.button("🔍 Klasifikasi Sekarang", type="primary"):
        if not validasi_input(teks_asli):
            st.warning("⚠️ Input tidak boleh kosong.")
            return

        # Jalankan pipeline langkah demi langkah
        teks_bersih = cleaning_pipeline(teks_asli)
        teks_nostop = stopword_remover.remove(teks_bersih)
        teks_stem   = stemmer.stem(teks_nostop)
        fitur_tfidf = vectorizer.transform([teks_stem])
        label_pred  = model_nb.predict(fitur_tfidf)[0]
        proba       = model_nb.predict_proba(fitur_tfidf)[0]
        kelas_nb    = model_nb.classes_

        prob_max = float(max(proba))

        st.markdown("---")
        st.subheader("Hasil Klasifikasi")

        col_badge, col_chart = st.columns([1, 2])
        with col_badge:
            st.markdown("**Label Prediksi:**")
            st.markdown(
                get_sentiment_badge_html(label_pred, prob_max),
                unsafe_allow_html=True,
            )
        with col_chart:
            df_proba = pd.DataFrame({"Kelas": kelas_nb, "Probabilitas": proba})
            fig_proba = px.bar(
                df_proba, x="Kelas", y="Probabilitas",
                color="Kelas", color_discrete_map=WARNA_SENTIMEN,
                text=df_proba["Probabilitas"].apply(lambda p: f"{p:.1%}"),
                range_y=[0, 1],
                title="Probabilitas per Kelas",
            )
            fig_proba.update_traces(textposition="outside")
            fig_proba.update_layout(showlegend=False)
            st.plotly_chart(fig_proba, use_container_width=True)

        with st.expander("🔬 Lihat langkah preprocessing step-by-step"):
            st.markdown("**Teks Asli:**")
            st.code(teks_asli, language=None)
            st.markdown("**Setelah `cleaning_pipeline`:**")
            st.code(teks_bersih, language=None)
            st.markdown("**Setelah penghapusan stopword (Sastrawi):**")
            st.code(teks_nostop, language=None)
            st.markdown("**Setelah stemming (Sastrawi):**")
            st.code(teks_stem, language=None)


# ------------------------------------------------------------------
# 5.5  page_pencarian
# ------------------------------------------------------------------
def page_pencarian() -> None:
    try:
        df_komentar, df_video, df_penceramah = load_data()
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memuat data: {e}")
        st.stop()

    st.title("🔍 Pencarian Penceramah & Sentimen Video")

    daftar_penceramah = sorted(df_penceramah[KOLOM_CHANNEL].dropna().unique())
    if not daftar_penceramah:
        st.warning("⚠️ Tidak ada data penceramah yang tersedia.")
        return

    penceramah_dipilih = st.selectbox(
        "Pilih Penceramah:",
        options=daftar_penceramah,
    )

    st.markdown("---")

    # --- Profil penceramah ---
    df_kom_p = df_komentar[df_komentar[KOLOM_CHANNEL] == penceramah_dipilih]
    df_vid_p = df_video[df_video[KOLOM_CHANNEL] == penceramah_dipilih].copy()

    # Hitung total video dari df_penceramah jika ada
    row_p = df_penceramah[df_penceramah[KOLOM_CHANNEL] == penceramah_dipilih]
    total_video_p    = int(row_p["total_video"].values[0]) if "total_video" in row_p.columns and not row_p.empty else len(df_vid_p)
    total_komentar_p = len(df_kom_p)

    sentimen_dist = df_kom_p[KOLOM_LABEL].value_counts().reset_index()
    sentimen_dist.columns = ["Sentimen", "Jumlah"]
    sentimen_dominan = sentimen_dist.iloc[0]["Sentimen"] if not sentimen_dist.empty else "N/A"

    st.subheader(f"👤 Profil: {penceramah_dipilih}")
    col_stat, col_donut = st.columns([1, 2])

    with col_stat:
        st.metric("🎬 Total Video",    str(total_video_p))
        st.metric("💬 Total Komentar", f"{total_komentar_p:,}")
        st.markdown("**Sentimen Dominan:**")
        st.markdown(get_sentiment_badge_html(sentimen_dominan), unsafe_allow_html=True)

    with col_donut:
        if not sentimen_dist.empty:
            fig_donut = px.pie(
                sentimen_dist, names="Sentimen", values="Jumlah",
                hole=0.5,
                color="Sentimen", color_discrete_map=WARNA_SENTIMEN,
                title="Distribusi Sentimen",
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")

    # --- Daftar video ---
    st.subheader("📋 Daftar Video")

    if df_vid_p.empty:
        st.info("ℹ️ Tidak ada data video untuk penceramah ini.")
        return

    # Selectbox pilih video
    label_options = []
    for i, row in df_vid_p.reset_index(drop=True).iterrows():
        judul = row.get("title") if pd.notna(row.get("title", None)) else f"Video {i+1}"
        label_options.append(str(judul))

    video_terpilih_label = st.selectbox("Pilih Video:", options=label_options)
    idx_video = label_options.index(video_terpilih_label)
    row_video = df_vid_p.reset_index(drop=True).iloc[idx_video]

    # Info video
    video_id    = row_video.get(KOLOM_VIDEO_ID, None)
    judul_video = row_video.get("title", "N/A")
    topik_video = row_video.get(KOLOM_TOPIK, "N/A")
    tanggal     = row_video.get("published_at", "N/A")
    ringkasan   = row_video.get("ringkasan_konten", "N/A")

    # Kartu info video
    info_cols = st.columns([2, 1])
    with info_cols[0]:
        st.markdown(f"**Judul:** {judul_video}")
        st.markdown(f"**Topik:** {topik_video} &nbsp;|&nbsp; **Tanggal:** {tanggal}")
        st.markdown(f"**Ringkasan:** {ringkasan}")
    with info_cols[1]:
        # Filter komentar video ini
        video_id_valid = (
            video_id and
            not (isinstance(video_id, float) and pd.isna(video_id)) and
            str(video_id).strip() != ""
        )
        if video_id_valid:
            yt_url = f"https://www.youtube.com/watch?v={video_id}"
            st.markdown(f"[▶️ Tonton di YouTube]({yt_url})")
            df_kom_v = df_komentar[df_komentar[KOLOM_VIDEO_ID] == video_id]
        else:
            st.info("ℹ️ Menggunakan filter channel + topik karena video_id tidak tersedia.")
            df_kom_v = df_komentar[
                (df_komentar[KOLOM_CHANNEL] == penceramah_dipilih) &
                (df_komentar[KOLOM_TOPIK] == topik_video)
            ]
            yt_url = None

        # Mini chart sentimen
        if not df_kom_v.empty:
            dist_v = df_kom_v[KOLOM_LABEL].value_counts().reset_index()
            dist_v.columns = ["Sentimen", "Jumlah"]
            fig_mini = px.bar(
                dist_v, x="Sentimen", y="Jumlah",
                color="Sentimen", color_discrete_map=WARNA_SENTIMEN,
                height=200, text="Jumlah",
            )
            fig_mini.update_traces(textposition="outside")
            fig_mini.update_layout(showlegend=False, margin=dict(t=20, b=20))
            st.plotly_chart(fig_mini, use_container_width=True)

    st.markdown("---")

    # --- Detail video ---
    st.subheader("🎬 Detail Video")

    if video_id_valid:
        st.video(f"https://www.youtube.com/watch?v={video_id}")
    else:
        st.info("ℹ️ Embed video tidak tersedia karena video_id tidak ditemukan.")

    if df_kom_v.empty:
        st.info("ℹ️ Tidak ada komentar untuk video ini.")
        return

    # Pie chart sentimen video
    dist_v2 = df_kom_v[KOLOM_LABEL].value_counts().reset_index()
    dist_v2.columns = ["Sentimen", "Jumlah"]
    fig_pie_v = px.pie(
        dist_v2, names="Sentimen", values="Jumlah",
        color="Sentimen", color_discrete_map=WARNA_SENTIMEN,
        title=f"Distribusi Sentimen — {judul_video}",
    )
    st.plotly_chart(fig_pie_v, use_container_width=True)

    # Tabel sampel komentar per kelas
    st.subheader("💬 Sampel Komentar per Kelas Sentimen")
    for kelas in ["Positif", "Negatif", "Netral"]:
        df_k = df_kom_v[df_kom_v[KOLOM_LABEL] == kelas]
        if df_k.empty:
            continue
        warna_hex = get_sentiment_color(kelas)
        with st.expander(f"{kelas} ({len(df_k)} komentar)", expanded=False):
            kolom_komentar = "comment_text" if "comment_text" in df_k.columns else KOLOM_KOMENTAR_BERSIH
            st.dataframe(
                df_k[[kolom_komentar]].head(20).rename(
                    columns={kolom_komentar: "Komentar"}
                ),
                use_container_width=True,
            )

    # Word cloud opsional
    teks_wc_v = " ".join(df_kom_v[KOLOM_KOMENTAR_BERSIH].dropna().astype(str))
    if teks_wc_v.strip():
        with st.expander("☁️ Word Cloud Komentar Video Ini"):
            _render_wordcloud(teks_wc_v, judul=judul_video)


# ---------------------------------------------------------------------------
# MAIN — Routing Sidebar
# ---------------------------------------------------------------------------

def main() -> None:
    pilihan = st.sidebar.radio(
        "Navigasi",
        options=[
            "Overview",
            "EDA Dashboard",
            "Evaluasi Model",
            "Demo Klasifikasi Real-Time",
            "Pencarian Penceramah & Sentimen Video",
        ],
    )

    if pilihan == "Overview":
        page_overview()
    elif pilihan == "EDA Dashboard":
        page_eda()
    elif pilihan == "Evaluasi Model":
        page_evaluasi()
    elif pilihan == "Demo Klasifikasi Real-Time":
        page_demo()
    elif pilihan == "Pencarian Penceramah & Sentimen Video":
        page_pencarian()


if __name__ == "__main__":
    main()
else:
    # Dipanggil langsung oleh Streamlit (bukan via __main__)
    main()
