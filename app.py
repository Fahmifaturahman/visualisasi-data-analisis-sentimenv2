"""
===========================================================================
Sistem Analisis Sentimen Konten Dakwah Digital — Kota Bandung
===========================================================================

CARA INSTALASI:
    pip install -r requirements.txt

CARA MENJALANKAN:
    streamlit run app.py
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
import json
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
# SECTION 1: Konfigurasi Halaman
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Analisis Sentimen Dakwah Digital — Bandung",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# SECTION 1B: Konstanta Global
# ---------------------------------------------------------------------------

NAMA_PENELITI = "Fahmi Faturahman"
NIM           = "1222008"

WARNA_SENTIMEN = {
    "Positif": "#10b981",
    "Negatif": "#ef4444",
}

# Warna brand
C_PRIMARY   = "#1e3a5f"   # navy
C_SECONDARY = "#2563eb"   # biru
C_ACCENT    = "#f59e0b"   # amber
C_BG        = "#f8fafc"
C_CARD      = "#ffffff"
C_TEXT      = "#1e293b"
C_MUTED     = "#64748b"

PATH_CSV_KOMENTAR      = "data/hasil_prediksi_lengkap.csv"
PATH_CSV_KOMENTAR_FINAL= "data/dataset_final_komentar.csv"
PATH_CSV_VIDEO         = "data/dataset_final_video.csv"
PATH_CSV_PENCERAMAH    = "data/dataset_final_per_penceramah.csv"
PATH_RINGKASAN         = "data/ringkasan_hasil.json"
PATH_MODEL_NB          = "model/model_naive_bayes.pkl"
PATH_VECTORIZER        = "model/tfidf_vectorizer.pkl"
DIR_EDA_REPORT         = "eda_report"

KOLOM_LABEL           = "sentiment_label"
KOLOM_PREDIKSI        = "label_prediksi"
KOLOM_SPLIT           = "split"
KOLOM_CHANNEL         = "channel_name"
KOLOM_TOPIK           = "topik_utama"
KOLOM_VIDEO_ID        = "video_id"
KOLOM_KOMENTAR_BERSIH = "comment_clean"
KOLOM_KOMENTAR_TEXT   = "comment_text"

KELAS_SENTIMEN = ["Positif", "Negatif"]

# ---------------------------------------------------------------------------
# SECTION 1C: Custom CSS
# ---------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        /* ── Import Font ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* ── Global ── */
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}
        .main .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }}

        /* ── Hide default Streamlit branding ── */
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        header {{ visibility: hidden; }}

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {C_PRIMARY} 0%, #0f2744 100%);
        }}
        [data-testid="stSidebar"] * {{
            color: #e2e8f0 !important;
        }}
        [data-testid="stSidebar"] .stRadio label {{
            padding: 10px 14px;
            border-radius: 8px;
            margin-bottom: 4px;
            display: block;
            transition: all 0.2s;
            font-size: 0.9rem;
            cursor: pointer;
        }}
        [data-testid="stSidebar"] .stRadio label:hover {{
            background: rgba(255,255,255,0.12) !important;
        }}
        [data-testid="stSidebar"] hr {{
            border-color: rgba(255,255,255,0.15) !important;
        }}

        /* ── Page hero banner ── */
        .page-hero {{
            background: linear-gradient(135deg, {C_PRIMARY} 0%, {C_SECONDARY} 100%);
            border-radius: 16px;
            padding: 36px 40px 32px;
            margin-bottom: 28px;
            color: white;
        }}
        .page-hero h1 {{
            font-size: 1.9rem;
            font-weight: 800;
            margin: 0 0 8px 0;
            line-height: 1.25;
            color: white !important;
        }}
        .page-hero p {{
            font-size: 0.95rem;
            opacity: 0.88;
            margin: 0;
            line-height: 1.6;
            color: #e2e8f0 !important;
        }}
        .page-hero .badge-row {{
            display: flex;
            gap: 10px;
            margin-top: 16px;
            flex-wrap: wrap;
        }}
        .page-hero .badge {{
            background: rgba(255,255,255,0.18);
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 20px;
            padding: 4px 14px;
            font-size: 0.8rem;
            font-weight: 500;
        }}

        /* ── Section heading ── */
        .section-heading {{
            font-size: 1.1rem;
            font-weight: 700;
            color: {C_TEXT};
            border-left: 4px solid {C_SECONDARY};
            padding-left: 12px;
            margin: 28px 0 16px 0;
            line-height: 1.3;
        }}

        /* ── KPI metric cards ── */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: {C_CARD};
            border-radius: 12px;
            padding: 20px 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
            border: 1px solid #e2e8f0;
            text-align: center;
            transition: box-shadow 0.2s;
        }}
        .kpi-card:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        }}
        .kpi-card .kpi-icon {{
            font-size: 1.6rem;
            margin-bottom: 6px;
        }}
        .kpi-card .kpi-value {{
            font-size: 1.65rem;
            font-weight: 800;
            color: {C_PRIMARY};
            line-height: 1.1;
        }}
        .kpi-card .kpi-label {{
            font-size: 0.75rem;
            color: {C_MUTED};
            font-weight: 500;
            margin-top: 4px;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}
        .kpi-card .kpi-delta {{
            font-size: 0.78rem;
            color: {C_SECONDARY};
            font-weight: 600;
            margin-top: 2px;
        }}

        /* ── Pipeline cards ── */
        .pipeline-wrap {{
            display: flex;
            align-items: stretch;
            gap: 0;
            margin-bottom: 8px;
        }}
        .pipeline-step {{
            flex: 1;
            background: {C_CARD};
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 24px 20px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }}
        .pipeline-step .step-num {{
            display: inline-block;
            width: 32px; height: 32px;
            border-radius: 50%;
            background: {C_SECONDARY};
            color: white;
            font-weight: 700;
            font-size: 0.9rem;
            line-height: 32px;
            margin-bottom: 10px;
        }}
        .pipeline-step h4 {{
            font-size: 1rem;
            font-weight: 700;
            color: {C_TEXT};
            margin: 0 0 6px 0;
        }}
        .pipeline-step p {{
            font-size: 0.82rem;
            color: {C_MUTED};
            line-height: 1.5;
            margin: 0;
        }}
        .pipeline-arrow {{
            display: flex;
            align-items: center;
            padding: 0 10px;
            font-size: 1.5rem;
            color: {C_MUTED};
        }}



        /* ── Sentiment badge ── */
        .badge-positif {{
            background: #dcfce7;
            color: #15803d;
            border: 1px solid #86efac;
            padding: 3px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }}
        .badge-negatif {{
            background: #fee2e2;
            color: #dc2626;
            border: 1px solid #fca5a5;
            padding: 3px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }}

        /* ── Info card ── */
        .info-card {{
            background: {C_CARD};
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px 22px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }}

        /* ── Demo result box ── */
        .result-positif {{
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border: 2px solid #10b981;
            border-radius: 14px;
            padding: 22px 24px;
            text-align: center;
        }}
        .result-negatif {{
            background: linear-gradient(135deg, #fff1f2 0%, #fee2e2 100%);
            border: 2px solid #ef4444;
            border-radius: 14px;
            padding: 22px 24px;
            text-align: center;
        }}
        .result-label {{
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            opacity: 0.7;
            margin-bottom: 6px;
        }}
        .result-sentiment {{
            font-size: 2rem;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 6px;
        }}
        .result-prob {{
            font-size: 1rem;
            font-weight: 600;
            opacity: 0.8;
        }}

        /* ── Penceramah profile card ── */
        .profile-card {{
            background: linear-gradient(135deg, {C_PRIMARY} 0%, {C_SECONDARY} 100%);
            border-radius: 14px;
            padding: 24px 26px;
            color: white;
            margin-bottom: 20px;
        }}
        .profile-card h3 {{
            font-size: 1.15rem;
            font-weight: 700;
            margin: 0 0 14px 0;
            color: white !important;
        }}
        .profile-card .stat-row {{
            display: flex;
            gap: 24px;
            flex-wrap: wrap;
        }}
        .profile-card .stat-item {{
            text-align: center;
        }}
        .profile-card .stat-val {{
            font-size: 1.5rem;
            font-weight: 800;
            display: block;
            line-height: 1;
        }}
        .profile-card .stat-lbl {{
            font-size: 0.72rem;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}

        /* ── Footer ── */
        .app-footer {{
            text-align: center;
            padding: 24px 0 8px;
            color: {C_MUTED};
            font-size: 0.8rem;
            border-top: 1px solid #e2e8f0;
            margin-top: 48px;
        }}

        /* ── Streamlit component tweaks ── */
        [data-testid="metric-container"] {{
            background: {C_CARD};
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 14px 16px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }}
        [data-testid="metric-container"] label {{
            font-size: 0.75rem !important;
            color: {C_MUTED} !important;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}
        [data-testid="metric-container"] [data-testid="stMetricValue"] {{
            font-size: 1.45rem !important;
            font-weight: 700 !important;
            color: {C_PRIMARY} !important;
        }}
        div[data-testid="stTabs"] button {{
            font-weight: 600;
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {C_SECONDARY} 0%, #1d4ed8 100%);
            border: none;
            border-radius: 8px;
            font-weight: 600;
            padding: 10px 28px;
            transition: all 0.2s;
        }}
        .stButton > button[kind="primary"]:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(37,99,235,0.4);
        }}
        .stTextArea textarea {{
            border-radius: 10px;
            border: 1.5px solid #e2e8f0;
            font-size: 0.92rem;
        }}
        .stTextArea textarea:focus {{
            border-color: {C_SECONDARY};
            box-shadow: 0 0 0 3px rgba(37,99,235,0.1);
        }}
        .stSelectbox > div > div {{
            border-radius: 8px;
            border: 1.5px solid #e2e8f0;
        }}
        .stExpander {{
            border: 1px solid #e2e8f0 !important;
            border-radius: 10px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# SECTION 2: Fungsi Cache
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> tuple:
    required_files = [PATH_CSV_KOMENTAR, PATH_CSV_VIDEO, PATH_CSV_PENCERAMAH]
    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File tidak ditemukan: {os.path.abspath(path)}")

    df_komentar   = pd.read_csv(PATH_CSV_KOMENTAR)
    df_video      = pd.read_csv(PATH_CSV_VIDEO)
    df_penceramah = pd.read_csv(PATH_CSV_PENCERAMAH)

    if KOLOM_VIDEO_ID not in df_video.columns:
        _raw_path = "data/raw/raw_data_video.csv"
        if os.path.exists(_raw_path):
            try:
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
            except Exception:
                pass

    return df_komentar, df_video, df_penceramah


@st.cache_data
def load_ringkasan() -> dict:
    if os.path.exists(PATH_RINGKASAN):
        with open(PATH_RINGKASAN, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_resource
def load_model() -> tuple:
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
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    except ImportError as e:
        raise ImportError("PySastrawi tidak terinstall. Jalankan: pip install PySastrawi") from e
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
        "[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251" "]+",
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
    t = hapus_url(teks)
    t = hapus_mention(t)
    t = hapus_emoji(t)
    t = hapus_nonalpha(t)
    t = case_folding(t)
    t = norm_spasi(t)
    return t


# ---------------------------------------------------------------------------
# SECTION 4: Helper UI
# ---------------------------------------------------------------------------

def get_sentiment_color(sentimen: str) -> str:
    return WARNA_SENTIMEN.get(sentimen, "#64748b")


def badge_html(sentimen: str, probabilitas: float = None) -> str:
    cls = "badge-positif" if sentimen == "Positif" else "badge-negatif"
    teks_prob = f" &nbsp;{probabilitas:.1%}" if probabilitas is not None else ""
    return f'<span class="{cls}">{sentimen}{teks_prob}</span>'


def validasi_input(teks: str) -> bool:
    return bool(teks.strip())


def _render_wordcloud(teks_gabung: str, judul: str = "", bg: str = "white") -> None:
    if not teks_gabung.strip():
        st.info("Tidak cukup teks untuk membuat word cloud.")
        return
    wc  = WordCloud(
        width=900, height=420,
        background_color=bg,
        colormap="viridis" if bg == "white" else "cool",
        max_words=200,
    ).generate(teks_gabung)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    if judul:
        ax.set_title(judul, fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    buf.seek(0)
    st.image(buf, use_container_width=True)


def section_heading(label: str) -> None:
    st.markdown(f'<div class="section-heading">{label}</div>', unsafe_allow_html=True)


def kpi_card(icon: str, value: str, label: str, delta: str = "") -> str:
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-label">{label}</div>
      {delta_html}
    </div>"""


def plotly_theme(fig, height: int = 380) -> go.Figure:
    """Apply consistent theme to all plotly figures."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color=C_TEXT),
        margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0")
    return fig


def footer() -> None:
    st.markdown(
        f"""
        <div class="app-footer">
            🕌 &nbsp; Sistem Analisis Sentimen Konten Dakwah Digital — Kota Bandung
            &nbsp;|&nbsp; Peneliti: <strong>{NAMA_PENELITI}</strong> &nbsp;|&nbsp; NIM: <strong>{NIM}</strong>
            &nbsp;|&nbsp; Model: Multinomial Naive Bayes + TF-IDF
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# SECTION 5: Halaman
# ---------------------------------------------------------------------------

# ------------------------------------------------------------------
# 5.1  page_overview
# ------------------------------------------------------------------
def page_overview() -> None:
    with st.spinner("Memuat data..."):
        try:
            df_komentar, _df_video, _df_penceramah = load_data()
        except FileNotFoundError as e:
            st.error(f"❌ {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memuat data: {e}")
            st.stop()

    ringkasan = load_ringkasan()

    # ── Hero Banner ──
    st.markdown(
        f"""
        <div class="page-hero">
          <h1>🕌 Analisis Sentimen Konten Dakwah Digital<br>Kota Bandung</h1>
          <p>
            Visualisasi hasil klasifikasi sentimen komentar YouTube dari penceramah dakwah Islam
            di Kota Bandung menggunakan <strong>Multinomial Naive Bayes</strong> dengan representasi fitur
            <strong>TF-IDF</strong> dan pelabelan berbasis <strong>Lexicon INSET</strong>.
          </p>
          <div class="badge-row">
            <span class="badge">👤 {NAMA_PENELITI}</span>
            <span class="badge">🎓 NIM {NIM}</span>
            <span class="badge">🤖 Multinomial Naive Bayes</span>
            <span class="badge">🎯 Akurasi 80.92%</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI Row 1 ──
    total_komentar    = ringkasan.get("total_data", len(df_komentar))
    jumlah_penceramah = df_komentar[KOLOM_CHANNEL].nunique()
    jumlah_topik      = df_komentar[KOLOM_TOPIK].nunique()
    akurasi           = ringkasan.get("accuracy", None)
    akurasi_str       = f"{akurasi:.2%}" if akurasi else "N/A"
    f1                = ringkasan.get("f1_score", None)
    f1_str            = f"{f1:.4f}" if f1 else "N/A"
    data_latih        = ringkasan.get("data_latih", "N/A")
    data_uji          = ringkasan.get("data_uji", "N/A")
    dist_json         = ringkasan.get("distribusi_label", {})
    jml_pos           = dist_json.get("Positif", 0)
    jml_neg           = dist_json.get("Negatif", 0)

    st.markdown(
        f"""
        <div class="kpi-grid">
          {kpi_card("💬", f"{total_komentar:,}", "Total Komentar Berlabel")}
          {kpi_card("🎙️", str(jumlah_penceramah), "Penceramah")}
          {kpi_card("📚", str(jumlah_topik), "Topik Dakwah")}
          {kpi_card("🎯", akurasi_str, "Akurasi Model MNB")}
        </div>
        <div class="kpi-grid">
          {kpi_card("⚖️", f1_str, "F1-Score (Weighted)")}
          {kpi_card("🏋️", f"{data_latih:,}" if isinstance(data_latih, int) else str(data_latih), "Data Latih (80%)")}
          {kpi_card("🧪", f"{data_uji:,}" if isinstance(data_uji, int) else str(data_uji), "Data Uji (20%)")}
          {kpi_card("✅", f"{jml_pos:,}", "Komentar Positif", f"{jml_pos/(jml_pos+jml_neg)*100:.1f}%" if (jml_pos+jml_neg)>0 else "")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Pipeline Analisis ──
    section_heading("📋 Pipeline Analisis")
    st.markdown(
        """
        <div class="pipeline-wrap">
          <div class="pipeline-step">
            <div class="step-num">1</div>
            <h4>📥 Scraping</h4>
            <p>Pengumpulan komentar YouTube via YouTube Data API v3 untuk penceramah dakwah Kota Bandung</p>
          </div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-step">
            <div class="step-num">2</div>
            <h4>🧹 Cleaning + EDA</h4>
            <p>Normalisasi teks, penghapusan stopword, stemming Sastrawi, Word2Vec embedding, dan eksplorasi data</p>
          </div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-step">
            <div class="step-num">3</div>
            <h4>🤖 Labeling + MNB</h4>
            <p>Pelabelan sentimen via Lexicon INSET, pelatihan Multinomial Naive Bayes dengan TF-IDF, dan evaluasi performa</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Distribusi Sentimen ──
    section_heading("📊 Distribusi Sentimen Keseluruhan")

    if dist_json:
        dist_df = pd.DataFrame(list(dist_json.items()), columns=["Sentimen", "Jumlah"])
    else:
        dist_df = df_komentar[KOLOM_LABEL].value_counts().reset_index()
        dist_df.columns = ["Sentimen", "Jumlah"]

    col_bar, col_pie = st.columns(2)
    with col_bar:
        fig_bar = px.bar(
            dist_df, x="Sentimen", y="Jumlah",
            color="Sentimen", color_discrete_map=WARNA_SENTIMEN,
            text="Jumlah",
        )
        fig_bar.update_traces(textposition="outside", textfont_size=13, textfont_color=C_TEXT)
        fig_bar.update_layout(showlegend=False, xaxis_title="", yaxis_title="Jumlah Komentar")
        st.plotly_chart(plotly_theme(fig_bar, 300), use_container_width=True)
    with col_pie:
        total_dist = dist_df["Jumlah"].sum()
        dist_df["pct"] = (dist_df["Jumlah"] / total_dist * 100).round(1)
        dist_df["label_full"] = dist_df.apply(
            lambda r: f"{r['Sentimen']}<br>{r['Jumlah']:,} ({r['pct']}%)", axis=1
        )
        fig_pie = px.pie(
            dist_df, names="Sentimen", values="Jumlah",
            color="Sentimen", color_discrete_map=WARNA_SENTIMEN,
            hole=0.52,
        )
        fig_pie.update_traces(
            texttemplate="<b>%{label}</b><br>%{value:,}<br>(%{percent})",
            textfont_size=11,
        )
        st.plotly_chart(plotly_theme(fig_pie, 300), use_container_width=True)

    footer()


# ------------------------------------------------------------------
# 5.2  page_eda
# ------------------------------------------------------------------
def page_eda() -> None:
    with st.spinner("Memuat data EDA..."):
        try:
            df_komentar, df_video, _df_penceramah = load_data()
        except FileNotFoundError as e:
            st.error(f"❌ {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ {e}")
            st.stop()

    # ── Hero ──
    st.markdown(
        """
        <div class="page-hero">
          <h1>📊 EDA Dashboard</h1>
          <p>Eksplorasi data interaktif — distribusi sentimen, pola penceramah, topik dakwah, dan analisis teks.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar Filter ──
    st.sidebar.markdown(
        '<div style="font-size:0.85rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;opacity:0.7;margin-bottom:8px;">Filter Data</div>',
        unsafe_allow_html=True,
    )
    filter_channel = st.sidebar.multiselect(
        "Penceramah", options=sorted(df_komentar[KOLOM_CHANNEL].dropna().unique()),
    )
    filter_topik = st.sidebar.multiselect(
        "Topik", options=sorted(df_komentar[KOLOM_TOPIK].dropna().unique()),
    )
    filter_sentimen = st.sidebar.multiselect(
        "Sentimen", options=KELAS_SENTIMEN, default=KELAS_SENTIMEN,
    )

    df_f = df_komentar.copy()
    if filter_channel:
        df_f = df_f[df_f[KOLOM_CHANNEL].isin(filter_channel)]
    if filter_topik:
        df_f = df_f[df_f[KOLOM_TOPIK].isin(filter_topik)]
    if filter_sentimen:
        df_f = df_f[df_f[KOLOM_LABEL].isin(filter_sentimen)]

    if df_f.empty:
        st.warning("⚠️ Tidak ada data yang sesuai dengan filter yang dipilih.")
        return

    st.markdown(
        f'<p style="color:{C_MUTED};font-size:0.85rem;margin-bottom:20px;">'
        f'Menampilkan <strong>{len(df_f):,}</strong> komentar dari <strong>{len(df_komentar):,}</strong> total.</p>',
        unsafe_allow_html=True,
    )

    # ── Tab-based layout ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Sentimen & Penceramah",
        "📁 Topik & Heatmap",
        "🔤 Analisis Teks",
        "☁️ Word Cloud",
        "🖼️ Grafik Notebook",
    ])

    # ── Tab 1: Sentimen & Penceramah ──
    with tab1:
        section_heading("Distribusi Sentimen")
        dist_sent = df_f[KOLOM_LABEL].value_counts().reset_index()
        dist_sent.columns = ["Sentimen", "Jumlah"]
        dist_sent["Persentase"] = (dist_sent["Jumlah"] / dist_sent["Jumlah"].sum() * 100).round(1)
        dist_sent["Teks"] = dist_sent.apply(lambda r: f"{r['Jumlah']:,} ({r['Persentase']}%)", axis=1)

        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.bar(
                dist_sent, x="Sentimen", y="Jumlah",
                color="Sentimen", color_discrete_map=WARNA_SENTIMEN, text="Teks",
            )
            fig1.update_traces(textposition="outside", textfont_size=12)
            fig1.update_layout(showlegend=False, xaxis_title="", yaxis_title="Jumlah")
            st.plotly_chart(plotly_theme(fig1, 320), use_container_width=True)
        with c2:
            fig1b = px.pie(
                dist_sent, names="Sentimen", values="Jumlah",
                color="Sentimen", color_discrete_map=WARNA_SENTIMEN, hole=0.45,
            )
            fig1b.update_traces(texttemplate="<b>%{label}</b><br>%{percent}", textfont_size=12)
            st.plotly_chart(plotly_theme(fig1b, 320), use_container_width=True)

        section_heading("Top 20 Penceramah — Jumlah Komentar")
        top_channel = df_f[KOLOM_CHANNEL].value_counts().head(20).reset_index()
        top_channel.columns = ["Penceramah", "Jumlah"]
        fig2 = px.bar(
            top_channel, x="Jumlah", y="Penceramah",
            orientation="h", text="Jumlah",
            color="Jumlah", color_continuous_scale=["#bfdbfe", C_SECONDARY, C_PRIMARY],
        )
        fig2.update_traces(textposition="outside", textfont_size=10)
        fig2.update_layout(
            yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
            xaxis_title="Jumlah Komentar", yaxis_title="",
        )
        st.plotly_chart(plotly_theme(fig2, max(360, len(top_channel) * 22)), use_container_width=True)

        section_heading("Komposisi Sentimen per Penceramah (Top 15)")
        top15 = df_f[KOLOM_CHANNEL].value_counts().head(15).index.tolist()
        pivot_sent = (
            df_f[df_f[KOLOM_CHANNEL].isin(top15)]
            .groupby([KOLOM_CHANNEL, KOLOM_LABEL])
            .size()
            .reset_index(name="Jumlah")
        )
        fig3 = px.bar(
            pivot_sent, x="Jumlah", y=KOLOM_CHANNEL,
            color=KOLOM_LABEL, color_discrete_map=WARNA_SENTIMEN,
            orientation="h", barmode="stack",
            labels={KOLOM_CHANNEL: "", "Jumlah": "Komentar"},
        )
        fig3.update_layout(yaxis=dict(autorange="reversed"), legend_title="Sentimen")
        st.plotly_chart(plotly_theme(fig3, max(360, len(top15) * 24)), use_container_width=True)

    # ── Tab 2: Topik & Heatmap ──
    with tab2:
        section_heading("Distribusi Topik Dakwah")
        dist_topik = df_f[KOLOM_TOPIK].value_counts().reset_index()
        dist_topik.columns = ["Topik", "Jumlah"]

        c_pie, c_bar = st.columns(2)
        with c_pie:
            fig4a = px.pie(
                dist_topik, names="Topik", values="Jumlah",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.3,
            )
            fig4a.update_traces(texttemplate="<b>%{label}</b><br>%{percent}", textfont_size=10)
            st.plotly_chart(plotly_theme(fig4a, 380), use_container_width=True)
        with c_bar:
            fig4b = px.bar(
                dist_topik, x="Jumlah", y="Topik",
                orientation="h", text="Jumlah",
                color="Jumlah", color_continuous_scale=["#e0f2fe", "#0ea5e9", "#0369a1"],
            )
            fig4b.update_traces(textposition="outside", textfont_size=10)
            fig4b.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(plotly_theme(fig4b, 380), use_container_width=True)

        section_heading("Heatmap Penceramah × Topik")
        pivot_data = (
            df_f.groupby([KOLOM_CHANNEL, KOLOM_TOPIK])
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
                    showscale=True,
                )
            )
            fig5.update_layout(
                xaxis_title="Topik", yaxis_title="",
                xaxis=dict(side="bottom"),
            )
            h = max(420, len(pivot_matrix) * 18)
            st.plotly_chart(plotly_theme(fig5, h), use_container_width=True)

    # ── Tab 3: Analisis Teks ──
    with tab3:
        section_heading("Top 30 Kata Paling Sering (Komentar Bersih)")
        teks_semua = " ".join(df_f[KOLOM_KOMENTAR_BERSIH].dropna().astype(str))
        ctr_all    = collections.Counter(teks_semua.split())
        top30      = ctr_all.most_common(30)
        if top30:
            df_top30 = pd.DataFrame(top30, columns=["Kata", "Frekuensi"])
            fig6 = px.bar(
                df_top30, x="Frekuensi", y="Kata",
                orientation="h", text="Frekuensi",
                color="Frekuensi",
                color_continuous_scale=["#d1fae5", "#10b981", "#065f46"],
            )
            fig6.update_traces(textposition="outside", textfont_size=10)
            fig6.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(plotly_theme(fig6, 700), use_container_width=True)

        section_heading("Kata Dominan per Sentimen")

        def top_kata(df, sentimen, n=15):
            teks = " ".join(df[df[KOLOM_LABEL] == sentimen][KOLOM_KOMENTAR_BERSIH].dropna().astype(str))
            ctr  = collections.Counter(teks.split())
            return pd.DataFrame(ctr.most_common(n), columns=["Kata", "Frekuensi"])

        c_pos, c_neg = st.columns(2)
        with c_pos:
            df_pk = top_kata(df_f, "Positif")
            fig7a = px.bar(
                df_pk, x="Frekuensi", y="Kata",
                orientation="h", text="Frekuensi",
                color_discrete_sequence=["#10b981"],
                title="🟢 Top 15 Kata — Positif",
            )
            fig7a.update_traces(textposition="outside", textfont_size=10)
            fig7a.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
            st.plotly_chart(plotly_theme(fig7a, 420), use_container_width=True)
        with c_neg:
            df_nk = top_kata(df_f, "Negatif")
            fig7b = px.bar(
                df_nk, x="Frekuensi", y="Kata",
                orientation="h", text="Frekuensi",
                color_discrete_sequence=["#ef4444"],
                title="🔴 Top 15 Kata — Negatif",
            )
            fig7b.update_traces(textposition="outside", textfont_size=10)
            fig7b.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
            st.plotly_chart(plotly_theme(fig7b, 420), use_container_width=True)

    # ── Tab 4: Word Cloud ──
    with tab4:
        section_heading("Word Cloud Komentar")
        wc1, wc2, wc3 = st.tabs(["🌐 Semua", "🟢 Positif", "🔴 Negatif"])
        with wc1:
            teks = " ".join(df_f[KOLOM_KOMENTAR_BERSIH].dropna().astype(str))
            _render_wordcloud(teks, "Semua Sentimen")
        with wc2:
            teks = " ".join(df_f[df_f[KOLOM_LABEL] == "Positif"][KOLOM_KOMENTAR_BERSIH].dropna().astype(str))
            _render_wordcloud(teks, "Komentar Positif")
        with wc3:
            teks = " ".join(df_f[df_f[KOLOM_LABEL] == "Negatif"][KOLOM_KOMENTAR_BERSIH].dropna().astype(str))
            _render_wordcloud(teks, "Komentar Negatif")

    # ── Tab 5: Grafik Notebook ──
    with tab5:
        png_files = sorted(Path(DIR_EDA_REPORT).glob("*.png")) if Path(DIR_EDA_REPORT).exists() else []
        if png_files:
            section_heading(f"Grafik dari Notebook ({len(png_files)} file)")
            cols = st.columns(2)
            for i, png in enumerate(png_files):
                with cols[i % 2]:
                    st.image(str(png), caption=png.name, use_container_width=True)
        else:
            st.info("ℹ️ Tidak ada file PNG di folder eda_report/.")

    footer()


# ------------------------------------------------------------------
# 5.3  page_evaluasi
# ------------------------------------------------------------------
def page_evaluasi() -> None:
    with st.spinner("Memuat data evaluasi..."):
        try:
            df_komentar, _df_video, _df_penceramah = load_data()
        except FileNotFoundError as e:
            st.error(f"❌ {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ {e}")
            st.stop()

    ringkasan = load_ringkasan()

    # ── Hero ──
    st.markdown(
        """
        <div class="page-hero">
          <h1>📈 Evaluasi Model</h1>
          <p>Performa Multinomial Naive Bayes dengan TF-IDF pada dataset komentar YouTube dakwah Kota Bandung.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Ringkasan Model ──
    if ringkasan:
        section_heading("📋 Ringkasan Model")
        st.markdown(
            f"""
            <div class="kpi-grid">
              {kpi_card("🤖", ringkasan.get("model", "MNB"), "Model")}
              {kpi_card("🔢", str(ringkasan.get("alpha", "N/A")), "Alpha (Laplace Smoothing)")}
              {kpi_card("📦", f"{ringkasan.get('total_data', 0):,}", "Total Data Berlabel")}
              {kpi_card("🔠", f"{ringkasan.get('jumlah_fitur', 5000):,}", "Fitur TF-IDF")}
            </div>
            <div class="kpi-grid">
              {kpi_card("🎯", f"{ringkasan.get('accuracy', 0):.4f}", "Accuracy")}
              {kpi_card("🔍", f"{ringkasan.get('precision', 0):.4f}", "Precision (Weighted)")}
              {kpi_card("📡", f"{ringkasan.get('recall', 0):.4f}", "Recall (Weighted)")}
              {kpi_card("⚖️", f"{ringkasan.get('f1_score', 0):.4f}", "F1-Score (Weighted)")}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Prep Data Test ──
    df_test = df_komentar[df_komentar[KOLOM_SPLIT] == "test"].copy()
    if df_test.empty:
        st.info("ℹ️ Data split test tidak tersedia.")
        return

    df_test = df_test.dropna(subset=[KOLOM_PREDIKSI])
    df_test = df_test[df_test[KOLOM_LABEL].isin(KELAS_SENTIMEN)]
    df_test = df_test[df_test[KOLOM_PREDIKSI].isin(KELAS_SENTIMEN)]

    if df_test.empty:
        st.info("ℹ️ Data test tidak cukup untuk evaluasi.")
        return

    y_true = df_test[KOLOM_LABEL]
    y_pred = df_test[KOLOM_PREDIKSI]
    cm     = confusion_matrix(y_true, y_pred, labels=KELAS_SENTIMEN)
    cm_norm= cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # ── Confusion Matrix ──
    section_heading("Confusion Matrix")
    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:
        st.caption("**Raw Count**")
        cm_colors = [[f"rgb(219,234,254)", f"rgb(239,68,68)"],
                     [f"rgb(239,68,68)", f"rgb(16,185,129)"]]
        fig_cm = go.Figure(
            go.Heatmap(
                z=cm, x=KELAS_SENTIMEN, y=KELAS_SENTIMEN,
                colorscale="Blues",
                text=cm, texttemplate="<b>%{text}</b>",
                textfont=dict(size=18),
                hoverongaps=False, showscale=False,
            )
        )
        fig_cm.update_layout(
            xaxis=dict(title="Prediksi", side="bottom"),
            yaxis=dict(title="Aktual"),
        )
        st.plotly_chart(plotly_theme(fig_cm, 320), use_container_width=True)

    with cm_col2:
        st.caption("**Normalized (per baris aktual)**")
        fig_cm_n = go.Figure(
            go.Heatmap(
                z=cm_norm, x=KELAS_SENTIMEN, y=KELAS_SENTIMEN,
                colorscale="RdYlGn",
                text=np.round(cm_norm, 3),
                texttemplate="<b>%{text}</b>",
                textfont=dict(size=18),
                hoverongaps=False, showscale=True,
                zmin=0, zmax=1,
            )
        )
        fig_cm_n.update_layout(
            xaxis=dict(title="Prediksi", side="bottom"),
            yaxis=dict(title="Aktual"),
        )
        st.plotly_chart(plotly_theme(fig_cm_n, 320), use_container_width=True)

    # ── Metrik per kelas ──
    section_heading("Metrik Evaluasi")
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Accuracy",         f"{accuracy:.4f}")
    m2.metric("🔍 Precision (w.avg)", f"{precision:.4f}")
    m3.metric("📡 Recall (w.avg)",    f"{recall:.4f}")
    m4.metric("⚖️ F1-Score (w.avg)",  f"{f1:.4f}")

    # ── Classification Report ──
    section_heading("Classification Report per Kelas")
    report_dict = classification_report(
        y_true, y_pred, labels=KELAS_SENTIMEN, output_dict=True, zero_division=0
    )
    df_report = pd.DataFrame(report_dict).T.round(4)
    st.dataframe(
        df_report.style
            .background_gradient(subset=["f1-score"], cmap="Greens")
            .format("{:.4f}", subset=["precision", "recall", "f1-score"]),
        use_container_width=True,
    )

    # ── Grafik dari notebook ──
    png_grafik = [
        "eda_report/grafik_evaluasi_naive_bayes.png",
        "eda_report/grafik_confusion_matrix_normalized.png",
        "eda_report/grafik_distribusi_label.png",
    ]
    existing = [p for p in png_grafik if os.path.exists(p)]
    if existing:
        section_heading("📊 Grafik dari Notebook")
        cols = st.columns(min(len(existing), 2))
        for i, p in enumerate(existing):
            with cols[i % 2]:
                st.image(p, caption=os.path.basename(p), use_container_width=True)

    # ── Tabel Misklasifikasi ──
    section_heading("Tabel Misklasifikasi (Prediksi Salah)")
    df_salah = df_test[df_test[KOLOM_LABEL] != df_test[KOLOM_PREDIKSI]].copy()
    if df_salah.empty:
        st.success("✅ Tidak ada prediksi yang salah pada data test!")
    else:
        kolom_tampil = [
            c for c in [KOLOM_KOMENTAR_TEXT, KOLOM_CHANNEL, KOLOM_TOPIK, KOLOM_LABEL, KOLOM_PREDIKSI]
            if c in df_salah.columns
        ]
        st.dataframe(
            df_salah[kolom_tampil].head(200)
                .style.apply(lambda row: ["background-color:#fef2f2"] * len(row), axis=1),
            use_container_width=True,
        )
        st.caption(f"Menampilkan 200 dari {len(df_salah):,} prediksi salah.")

    footer()


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
        st.error(f"❌ {e}")
        st.stop()

    # ── Hero ──
    st.markdown(
        """
        <div class="page-hero">
          <h1>🤖 Demo Klasifikasi Real-Time</h1>
          <p>Masukkan komentar berbahasa Indonesia tentang konten dakwah, model akan mengklasifikasikan sentimen secara langsung.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    CONTOH = [
        "— Pilih contoh komentar —",
        "Masya Allah ceramahnya sangat menyentuh hati, semoga Allah merahmati ustadz",
        "Penjelasannya mudah dipahami dan sangat bermanfaat untuk kehidupan sehari-hari",
        "Alhamdulillah ilmu yang sangat berharga, terima kasih sudah berbagi",
        "Konten yang sangat informatif dan inspiratif, terus berkarya ustadz",
        "Subhanallah, kajian yang dalam sekali, sangat membuka wawasan saya",
        "Suaranya tidak jelas dan materi yang disampaikan membingungkan",
        "Ceramah ini tidak sesuai dengan ajaran yang benar menurut saya",
        "Sangat mengecewakan, konten seperti ini menyesatkan umat",
    ]

    c_input, c_result = st.columns([3, 2])

    with c_input:
        section_heading("✍️ Input Komentar")
        pilihan = st.selectbox("Pilih contoh:", CONTOH)
        teks_default = "" if pilihan == CONTOH[0] else pilihan
        teks_asli = st.text_area(
            "Teks komentar:",
            value=teks_default,
            height=140,
            placeholder="Ketik atau tempel komentar berbahasa Indonesia...",
            label_visibility="collapsed",
        )
        submit = st.button("🔍 Klasifikasi Sekarang", type="primary", use_container_width=True)

    with c_result:
        section_heading("📋 Hasil")
        result_placeholder = st.empty()
        result_placeholder.markdown(
            '<div class="info-card" style="text-align:center;color:#94a3b8;padding:40px 20px;">'
            '<div style="font-size:2rem;margin-bottom:8px;">⬅️</div>'
            '<div style="font-size:0.9rem;">Masukkan komentar dan klik Klasifikasi</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    if submit:
        if not validasi_input(teks_asli):
            st.warning("⚠️ Input tidak boleh kosong.")
            return

        with st.spinner("Menganalisis..."):
            teks_bersih = cleaning_pipeline(teks_asli)
            teks_nostop = stopword_remover.remove(teks_bersih)
            teks_stem   = stemmer.stem(teks_nostop)
            fitur_tfidf = vectorizer.transform([teks_stem])
            label_pred  = model_nb.predict(fitur_tfidf)[0]
            proba       = model_nb.predict_proba(fitur_tfidf)[0]
            kelas_nb    = model_nb.classes_
            prob_max    = float(max(proba))

        is_pos  = label_pred == "Positif"
        res_cls = "result-positif" if is_pos else "result-negatif"
        res_col = "#10b981" if is_pos else "#ef4444"
        res_ico = "💚" if is_pos else "❤️"
        res_msg = "Apresiasi, pujian, atau dukungan terhadap konten dakwah." if is_pos else \
                  "Kritik, ketidaksetujuan, atau komentar tidak mendukung."

        with c_result:
            result_placeholder.markdown(
                f"""
                <div class="{res_cls}">
                  <div class="result-label">{res_ico} Prediksi Sentimen</div>
                  <div class="result-sentiment" style="color:{res_col};">{label_pred}</div>
                  <div class="result-prob">Keyakinan: {prob_max:.1%}</div>
                  <div style="font-size:0.78rem;margin-top:10px;opacity:0.75;">{res_msg}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Probabilitas chart ──
        section_heading("📊 Probabilitas per Kelas")
        df_proba = pd.DataFrame({"Kelas": kelas_nb, "Probabilitas": proba})
        fig_proba = px.bar(
            df_proba, x="Kelas", y="Probabilitas",
            color="Kelas", color_discrete_map=WARNA_SENTIMEN,
            text=df_proba["Probabilitas"].apply(lambda p: f"{p:.1%}"),
            range_y=[0, 1],
        )
        fig_proba.update_traces(textposition="outside", textfont_size=14)
        fig_proba.update_layout(showlegend=False, xaxis_title="", yaxis_title="Probabilitas")
        st.plotly_chart(plotly_theme(fig_proba, 280), use_container_width=True)

        # ── Preprocessing steps ──
        section_heading("🔬 Langkah Preprocessing")
        steps = [
            ("1. Teks Asli", teks_asli),
            ("2. Setelah Cleaning (hapus URL, mention, emoji, non-alpha, lowercase)", teks_bersih),
            ("3. Setelah Stopword Removal (Sastrawi)", teks_nostop),
            ("4. Setelah Stemming (Sastrawi)", teks_stem),
        ]
        for step_label, step_text in steps:
            with st.expander(step_label):
                st.code(step_text, language=None)

    footer()


# ------------------------------------------------------------------
# 5.5  page_pencarian
# ------------------------------------------------------------------
def page_pencarian() -> None:
    with st.spinner("Memuat data..."):
        try:
            df_komentar, df_video, df_penceramah = load_data()
        except FileNotFoundError as e:
            st.error(f"❌ {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ {e}")
            st.stop()

    # ── Hero ──
    st.markdown(
        """
        <div class="page-hero">
          <h1>🔍 Pencarian Penceramah &amp; Video</h1>
          <p>Jelajahi profil sentimen setiap penceramah, lihat distribusi komentar, dan detail tiap video.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    daftar_penceramah = (
        sorted(df_penceramah[KOLOM_CHANNEL].dropna().unique())
        if KOLOM_CHANNEL in df_penceramah.columns
        else sorted(df_komentar[KOLOM_CHANNEL].dropna().unique())
    )

    if not daftar_penceramah:
        st.warning("⚠️ Tidak ada data penceramah.")
        return

    penceramah_dipilih = st.selectbox("🎙️ Pilih Penceramah:", options=daftar_penceramah)

    # ── Data penceramah ──
    df_kom_p = df_komentar[df_komentar[KOLOM_CHANNEL] == penceramah_dipilih]
    df_vid_p = df_video[df_video[KOLOM_CHANNEL] == penceramah_dipilih].copy()
    row_p    = df_penceramah[df_penceramah[KOLOM_CHANNEL] == penceramah_dipilih]

    total_k  = int(row_p["total_komentar"].values[0]) if "total_komentar" in row_p.columns and not row_p.empty else len(df_kom_p)
    jml_pos  = int(row_p["jumlah_positif"].values[0]) if "jumlah_positif" in row_p.columns and not row_p.empty else len(df_kom_p[df_kom_p[KOLOM_LABEL] == "Positif"])
    jml_neg  = int(row_p["jumlah_negatif"].values[0]) if "jumlah_negatif" in row_p.columns and not row_p.empty else len(df_kom_p[df_kom_p[KOLOM_LABEL] == "Negatif"])
    pct_pos  = float(row_p["pct_positif"].values[0]) if "pct_positif" in row_p.columns and not row_p.empty else (jml_pos / total_k * 100 if total_k > 0 else 0)
    pct_neg  = float(row_p["pct_negatif"].values[0]) if "pct_negatif" in row_p.columns and not row_p.empty else (jml_neg / total_k * 100 if total_k > 0 else 0)
    dom      = row_p["sentimen_dominan"].values[0] if "sentimen_dominan" in row_p.columns and not row_p.empty else ("Positif" if jml_pos >= jml_neg else "Negatif")

    # ── Profil card ──
    dom_badge = badge_html(str(dom))
    st.markdown(
        f"""
        <div class="profile-card">
          <h3>👤 {penceramah_dipilih}</h3>
          <div class="stat-row">
            <div class="stat-item">
              <span class="stat-val">{total_k:,}</span>
              <span class="stat-lbl">Total Komentar</span>
            </div>
            <div class="stat-item">
              <span class="stat-val">{len(df_vid_p)}</span>
              <span class="stat-lbl">Video</span>
            </div>
            <div class="stat-item">
              <span class="stat-val" style="color:#a7f3d0;">{jml_pos:,}</span>
              <span class="stat-lbl">Positif ({pct_pos:.1f}%)</span>
            </div>
            <div class="stat-item">
              <span class="stat-val" style="color:#fca5a5;">{jml_neg:,}</span>
              <span class="stat-lbl">Negatif ({pct_neg:.1f}%)</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Donut & bar sentimen
    c_donut, c_bar_vid = st.columns([1, 2])
    with c_donut:
        if jml_pos > 0 or jml_neg > 0:
            fig_d = px.pie(
                pd.DataFrame({"Sentimen": ["Positif", "Negatif"], "Jumlah": [jml_pos, jml_neg]}),
                names="Sentimen", values="Jumlah",
                color="Sentimen", color_discrete_map=WARNA_SENTIMEN, hole=0.55,
            )
            fig_d.update_traces(texttemplate="<b>%{label}</b><br>%{percent}", textfont_size=11)
            st.plotly_chart(plotly_theme(fig_d, 280), use_container_width=True)

    with c_bar_vid:
        if not df_vid_p.empty and "pct_positif" in df_vid_p.columns:
            df_vid_sorted = df_vid_p.dropna(subset=["pct_positif"]).sort_values("pct_positif", ascending=True)
            fig_vb = px.bar(
                df_vid_sorted, x="pct_positif", y="title",
                orientation="h",
                color="pct_positif",
                color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
                range_x=[0, 100],
                labels={"pct_positif": "% Positif", "title": ""},
                title="% Sentimen Positif per Video",
            )
            fig_vb.update_layout(coloraxis_showscale=False, yaxis=dict(tickfont=dict(size=9)))
            h = max(280, len(df_vid_sorted) * 22)
            st.plotly_chart(plotly_theme(fig_vb, h), use_container_width=True)

    st.markdown("---")

    # ── Pilih Video ──
    if df_vid_p.empty:
        st.info("ℹ️ Tidak ada data video untuk penceramah ini.")
        footer()
        return

    section_heading("🎬 Detail Video")

    # Tabel video
    kolom_v_display = [c for c in [
        "title", KOLOM_TOPIK, "published_at",
        "total_komentar", "Positif", "Negatif",
        "pct_positif", "sentimen_dominan"
    ] if c in df_vid_p.columns]
    with st.expander("📊 Tabel Semua Video Penceramah Ini"):
        st.dataframe(df_vid_p[kolom_v_display].reset_index(drop=True), use_container_width=True)

    label_opts = [
        str(row.get("title", f"Video {i+1}"))
        for i, (_, row) in enumerate(df_vid_p.reset_index(drop=True).iterrows())
    ]
    sel_label = st.selectbox("Pilih Video:", options=label_opts)
    idx_v     = label_opts.index(sel_label)
    row_v     = df_vid_p.reset_index(drop=True).iloc[idx_v]

    video_id    = row_v.get(KOLOM_VIDEO_ID, None)
    judul_v     = row_v.get("title", "N/A")
    topik_v     = row_v.get(KOLOM_TOPIK, "N/A")
    tanggal_v   = row_v.get("published_at", "N/A")
    ringkasan_v = row_v.get("ringkasan_konten", "")
    kesimpulan_v= row_v.get("kesimpulan", "")
    vid_pos     = row_v.get("Positif", None)
    vid_neg     = row_v.get("Negatif", None)
    vid_total   = row_v.get("total_komentar", None)
    vid_pct_pos = row_v.get("pct_positif", None)
    vid_pct_neg = row_v.get("pct_negatif", None)
    vid_dom     = row_v.get("sentimen_dominan", None)

    vid_ok = (
        video_id is not None and
        not (isinstance(video_id, float) and pd.isna(video_id)) and
        str(video_id).strip() != ""
    )

    # Video info card
    yt_link = f'<a href="https://www.youtube.com/watch?v={video_id}" target="_blank">▶️ Tonton di YouTube</a>' if vid_ok else ""
    st.markdown(
        f"""
        <div class="info-card">
          <div style="font-size:1.05rem;font-weight:700;color:{C_TEXT};margin-bottom:8px;">{judul_v}</div>
          <div style="font-size:0.83rem;color:{C_MUTED};margin-bottom:10px;">
            📁 {topik_v} &nbsp;|&nbsp; 📅 {tanggal_v} &nbsp;|&nbsp; {yt_link}
          </div>
          {"<p style='font-size:0.85rem;color:" + C_TEXT + ";'><strong>Ringkasan:</strong> " + str(ringkasan_v) + "</p>" if ringkasan_v else ""}
          {"<p style='font-size:0.85rem;color:" + C_TEXT + ";margin-top:4px;'><strong>Kesimpulan:</strong> " + str(kesimpulan_v) + "</p>" if kesimpulan_v else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metrics video
    if vid_total is not None and not (isinstance(vid_total, float) and pd.isna(vid_total)):
        mv1, mv2, mv3, mv4 = st.columns(4)
        mv1.metric("💬 Total Komentar", f"{int(vid_total):,}")
        if vid_pos is not None and not (isinstance(vid_pos, float) and pd.isna(vid_pos)):
            mv2.metric("✅ Positif", f"{int(vid_pos):,}", f"{vid_pct_pos:.1f}%" if vid_pct_pos else None)
        if vid_neg is not None and not (isinstance(vid_neg, float) and pd.isna(vid_neg)):
            mv3.metric("❌ Negatif", f"{int(vid_neg):,}", f"{vid_pct_neg:.1f}%" if vid_pct_neg else None, delta_color="inverse")
        if vid_dom:
            mv4.metric("🏆 Dominan", str(vid_dom))

    # Embed YouTube
    if vid_ok:
        st.video(f"https://www.youtube.com/watch?v={video_id}")

    # Komentar
    df_kom_v = (
        df_komentar[df_komentar[KOLOM_VIDEO_ID] == video_id]
        if vid_ok
        else df_komentar[
            (df_komentar[KOLOM_CHANNEL] == penceramah_dipilih) &
            (df_komentar[KOLOM_TOPIK] == topik_v)
        ]
    )

    if df_kom_v.empty:
        st.info("ℹ️ Tidak ada komentar untuk video ini.")
        footer()
        return

    c_pie_v, c_wc_v = st.columns(2)
    with c_pie_v:
        dist_v = df_kom_v[KOLOM_LABEL].value_counts().reset_index()
        dist_v.columns = ["Sentimen", "Jumlah"]
        fig_pv = px.pie(
            dist_v, names="Sentimen", values="Jumlah",
            color="Sentimen", color_discrete_map=WARNA_SENTIMEN, hole=0.45,
        )
        fig_pv.update_traces(texttemplate="<b>%{label}</b><br>%{percent}", textfont_size=12)
        st.plotly_chart(plotly_theme(fig_pv, 280), use_container_width=True)
    with c_wc_v:
        teks_v = " ".join(df_kom_v[KOLOM_KOMENTAR_BERSIH].dropna().astype(str))
        if teks_v.strip():
            _render_wordcloud(teks_v)

    # Sampel komentar
    section_heading("💬 Sampel Komentar per Sentimen")
    for kelas in KELAS_SENTIMEN:
        df_k = df_kom_v[df_kom_v[KOLOM_LABEL] == kelas]
        if df_k.empty:
            continue
        col_k = KOLOM_KOMENTAR_TEXT if KOLOM_KOMENTAR_TEXT in df_k.columns else KOLOM_KOMENTAR_BERSIH
        with st.expander(f"{'🟢' if kelas=='Positif' else '🔴'} {kelas} — {len(df_k):,} komentar"):
            st.dataframe(
                df_k[[col_k]].head(30).rename(columns={col_k: "Komentar"}),
                use_container_width=True,
            )

    footer()


# ------------------------------------------------------------------
# 5.6  page_ranking_penceramah
# ------------------------------------------------------------------
def page_ranking_penceramah() -> None:
    with st.spinner("Memuat data ranking..."):
        try:
            df_komentar, df_video, df_penceramah = load_data()
        except FileNotFoundError as e:
            st.error(f"❌ {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ {e}")
            st.stop()

    # ── Hero ──
    st.markdown(
        """
        <div class="page-hero">
          <h1>🏆 Ranking Penceramah</h1>
          <p>Peringkat penceramah berdasarkan persentase komentar positif, volume komentar, dan sentimen dominan.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df_penceramah.empty:
        st.warning("⚠️ Data penceramah tidak tersedia.")
        return

    df_p = df_penceramah.copy()
    if df_p.columns[0] in ["", "Unnamed: 0"]:
        df_p = df_p.drop(columns=[df_p.columns[0]])

    if "pct_positif" in df_p.columns:
        df_p = df_p.sort_values("pct_positif", ascending=False).reset_index(drop=True)
        df_p.index += 1

    # ── Top 3 highlight ──
    top3 = df_p.head(3)
    medals = ["🥇", "🥈", "🥉"]
    section_heading("🎖️ Top 3 Penceramah Sentimen Positif Tertinggi")
    cols3 = st.columns(3)
    for i, (_, row) in enumerate(top3.iterrows()):
        with cols3[i]:
            pct = row.get("pct_positif", 0)
            nama = row.get(KOLOM_CHANNEL, "N/A")
            total = row.get("total_komentar", 0)
            st.markdown(
                f"""
                <div class="info-card" style="text-align:center;padding:24px 16px;">
                  <div style="font-size:2.2rem;">{medals[i]}</div>
                  <div style="font-weight:700;font-size:0.9rem;color:{C_TEXT};margin:8px 0 4px;">{nama}</div>
                  <div style="font-size:1.8rem;font-weight:800;color:#10b981;">{pct:.1f}%</div>
                  <div style="font-size:0.75rem;color:{C_MUTED};">Positif dari {int(total):,} komentar</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Tabel lengkap ──
    section_heading("📊 Tabel Lengkap Ranking")
    kolom_tampil = [c for c in [
        KOLOM_CHANNEL, "total_komentar", "jumlah_positif", "jumlah_negatif",
        "pct_positif", "pct_negatif", "sentimen_dominan"
    ] if c in df_p.columns]

    st.dataframe(
        df_p[kolom_tampil]
            .style.background_gradient(subset=["pct_positif"] if "pct_positif" in kolom_tampil else [], cmap="Greens")
            .format({c: "{:.1f}%" for c in ["pct_positif", "pct_negatif"] if c in kolom_tampil}),
        use_container_width=True,
        height=460,
    )

    # ── Bar chart % positif ──
    if "pct_positif" in df_p.columns:
        section_heading("📈 % Komentar Positif (Top 20 Penceramah)")
        df_top20 = df_p.head(20).copy()
        fig_rank = px.bar(
            df_top20, x="pct_positif", y=KOLOM_CHANNEL,
            orientation="h",
            text=df_top20["pct_positif"].apply(lambda x: f"{x:.1f}%"),
            color="pct_positif",
            color_continuous_scale=["#fca5a5", "#fbbf24", "#34d399"],
            range_x=[0, 100],
            labels={"pct_positif": "% Positif", KOLOM_CHANNEL: ""},
        )
        fig_rank.update_traces(textposition="outside", textfont_size=10)
        fig_rank.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(plotly_theme(fig_rank, max(400, len(df_top20) * 24)), use_container_width=True)

    # ── Scatter ──
    if "pct_positif" in df_p.columns and "total_komentar" in df_p.columns:
        section_heading("🔵 Scatter: Volume Komentar vs % Sentimen Positif")
        fig_sc = px.scatter(
            df_p, x="total_komentar", y="pct_positif",
            text=KOLOM_CHANNEL,
            color="sentimen_dominan" if "sentimen_dominan" in df_p.columns else None,
            color_discrete_map=WARNA_SENTIMEN,
            size="total_komentar",
            size_max=40,
            labels={"total_komentar": "Total Komentar", "pct_positif": "% Positif"},
        )
        fig_sc.update_traces(textposition="top center", textfont_size=9)
        fig_sc.update_layout(legend_title="Sentimen Dominan")
        st.plotly_chart(plotly_theme(fig_sc, 500), use_container_width=True)

    footer()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    inject_css()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown(
            f"""
            <div style="text-align:center;padding:20px 0 16px;">
              <div style="font-size:2.4rem;">🕌</div>
              <div style="font-size:0.95rem;font-weight:700;color:#f8fafc;line-height:1.3;margin-top:6px;">
                Sentimen Dakwah<br>Digital Bandung
              </div>
              <div style="font-size:0.75rem;opacity:0.65;margin-top:4px;">{NAMA_PENELITI} · {NIM}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<hr style="border-color:rgba(255,255,255,0.15);margin:0 0 12px;">', unsafe_allow_html=True)

        menu_items = {
            "🏠  Overview":                    "Overview",
            "📊  EDA Dashboard":               "EDA Dashboard",
            "📈  Evaluasi Model":              "Evaluasi Model",
            "🤖  Demo Klasifikasi":            "Demo Klasifikasi Real-Time",
            "🔍  Pencarian Penceramah":        "Pencarian Penceramah & Video",
            "🏆  Ranking Penceramah":          "Ranking Penceramah",
        }

        pilihan_label = st.radio(
            "Navigasi",
            options=list(menu_items.keys()),
            label_visibility="collapsed",
        )
        pilihan = menu_items[pilihan_label]

        st.markdown(
            f"""
            <div style="position:absolute;bottom:20px;left:0;right:0;text-align:center;
                        font-size:0.72rem;opacity:0.4;color:white;">
              MNB · TF-IDF 5K · Akurasi 80.92%
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Route ──
    if pilihan == "Overview":
        page_overview()
    elif pilihan == "EDA Dashboard":
        page_eda()
    elif pilihan == "Evaluasi Model":
        page_evaluasi()
    elif pilihan == "Demo Klasifikasi Real-Time":
        page_demo()
    elif pilihan == "Pencarian Penceramah & Video":
        page_pencarian()
    elif pilihan == "Ranking Penceramah":
        page_ranking_penceramah()


if __name__ == "__main__":
    main()
else:
    main()
