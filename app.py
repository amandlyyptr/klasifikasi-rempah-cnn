import streamlit as st
import os
import json
import zipfile
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from PIL import Image
import io
import time
import random
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Rempah CNN",
    page_icon=":seedling:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CONSTANTS ──────────────────────────────────────────────────────────────────
BASE_DIR    = "hasil_rempah"
DATASET_DIR = f"{BASE_DIR}/dataset"
MODEL_DIR   = f"{BASE_DIR}/model"
LAPORAN_DIR = f"{BASE_DIR}/laporan"
GRAFIK_DIR  = f"{BASE_DIR}/grafik"

for d in [DATASET_DIR, MODEL_DIR, LAPORAN_DIR, GRAFIK_DIR]:
    os.makedirs(d, exist_ok=True)

STATE_FILE = f"{BASE_DIR}/state.json"

# ─── SESSION STATE ──────────────────────────────────────────────────────────────
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            data = json.load(f)
        for k, v in data.items():
            if k not in st.session_state:
                st.session_state[k] = v

def save_state():
    keys = ["dataset_ready", "split_ready", "aug_config",
            "training_history", "eval_results", "class_names", "num_classes"]
    data = {k: st.session_state[k] for k in keys if k in st.session_state}
    with open(STATE_FILE, "w") as f:
        json.dump(data, f)

load_state()

def get_state(key, default=None):
    return st.session_state.get(key, default)

def set_state(key, value):
    st.session_state[key] = value
    save_state()

# ─── HELPER: Bersihkan dan validasi DataFrame CSV ───────────────────────────────
def clean_csv_dataframe(df):
    """Bersihkan kolom CSV dari BOM, spasi, dan case; rebuild label jika perlu."""
    # Selalu buat salinan eksplisit agar tidak ada referensi shared
    df = df.copy(deep=True)

    # Hapus BOM + strip + lowercase pada nama kolom
    new_cols = []
    for c in df.columns:
        c = c.replace('\ufeff', '').replace('\ufffe', '').strip().lower()
        new_cols.append(c)
    df.columns = new_cols

    # Rename kolom path jika namanya bukan 'filepath'
    if 'filepath' not in df.columns:
        path_candidates = [
            c for c in df.columns
            if any(k in c for k in ('path', 'file', 'img', 'image', 'gambar'))
        ]
        if path_candidates:
            df = df.rename(columns={path_candidates[0]: 'filepath'})

    # Pastikan kolom filepath ada
    if 'filepath' not in df.columns:
        raise ValueError(
            f"Kolom 'filepath' tidak ditemukan. Kolom tersedia: {df.columns.tolist()}"
        )

    # Pastikan tipe string dulu sebelum extract label
    df['filepath'] = df['filepath'].astype(str).str.strip()

    # Rebuild kolom label dari path jika tidak ada atau semua kosong
    if 'label' not in df.columns or df['label'].astype(str).str.strip().eq('').all():
        df['label'] = df['filepath'].apply(
            lambda x: os.path.basename(os.path.dirname(x))
        )

    df['label'] = df['label'].astype(str).str.strip()

    # Buang baris yang label atau filepath-nya kosong/NaN
    df = df[df['filepath'].str.len() > 0]
    df = df[df['label'].str.len() > 0]
    df = df.reset_index(drop=True)

    return df

# ─── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Font ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #f0f9ff;
    border-right: 1px solid #bae6fd;
}
[data-testid="stSidebar"] * {
    color: #0c3547 !important;
}

/* Nav radio — kotak seragam */
[data-testid="stSidebar"] .stRadio > div {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    background: #ffffff !important;
    border: 1.5px solid #bae6fd !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: #0c3547 !important;
    cursor: pointer;
    transition: all 0.15s ease;
    min-height: 44px;
    width: 100%;
    box-sizing: border-box;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: #e0f2fe !important;
    border-color: #0ea5e9 !important;
}
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + label,
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: #2e7d32 !important;
    border-color: #1b5e20 !important;
    color: #ffffff !important;
}

/* Status kotak seragam */
.status-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    margin-top: 8px;
}
.status-item {
    background: #ffffff;
    border: 1.5px solid #bae6fd;
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 0.78rem;
    font-weight: 500;
    color: #0c3547;
    display: flex;
    align-items: center;
    gap: 6px;
    min-height: 38px;
}
.status-item.done {
    background: #e8f5e9;
    border-color: #2e7d32;
    color: #1b5e20;
}
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #7dd3fc;
    flex-shrink: 0;
}
.status-item.done .status-dot {
    background: #2e7d32;
}

/* ── Metric Card ── */
.metric-card {
    background: linear-gradient(135deg, #0ea5e9, #2e7d32);
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    color: white;
    margin: 4px 0;
    box-shadow: 0 2px 8px rgba(14,165,233,0.22);
}
.metric-card .val {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
}
.metric-card .lbl {
    font-size: 0.82rem;
    opacity: 0.85;
    margin-top: 4px;
}

/* ── Step Badge ── */
.step-badge {
    display: inline-block;
    background: #e0f2fe;
    color: #0369a1;
    border: 1.5px solid #7dd3fc;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 10px;
}

/* ── Info / Warn Box ── */
.info-box {
    background: #f0f9ff;
    border-left: 4px solid #0ea5e9;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.9rem;
    color: #0c3547;
}
.warn-box {
    background: #fff8e1;
    border-left: 4px solid #f9a825;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.9rem;
    color: #4a3000;
}

/* ── Alur Step ── */
.flow-step {
    display: flex;
    align-items: flex-start;
    margin: 8px 0;
    padding: 12px 14px;
    background: #f0f9ff;
    border-radius: 10px;
    border: 1px solid #bae6fd;
    gap: 12px;
}
.flow-num {
    background: linear-gradient(135deg, #0ea5e9, #2e7d32);
    color: white;
    border-radius: 50%;
    width: 28px; height: 28px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700;
    font-size: 0.85rem;
    flex-shrink: 0;
}
.flow-title { font-weight: 600; color: #0c3547; font-size: 0.92rem; }
.flow-desc  { color: #555; font-size: 0.83rem; margin-top: 2px; }

/* ── Expander ── */
div[data-testid="stExpander"] {
    border: 1.5px solid #bae6fd !important;
    border-radius: 10px !important;
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: #2e7d32 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1b5e20 !important;
}

/* ── Sidebar brand ── */
.sidebar-brand {
    padding: 12px 0 4px 0;
    font-size: 1.15rem;
    font-weight: 700;
    color: #0369a1;
    letter-spacing: -0.3px;
}
.sidebar-sub {
    font-size: 0.78rem;
    color: #0ea5e9;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">Klasifikasi Rempah</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">CNN Transfer Learning</div>', unsafe_allow_html=True)
    st.divider()

    pages = [
        "Home",
        "Dataset",
        "Preprocessing & Augmentasi",
        "Training",
        "Riwayat Training",
        "Evaluasi",
        "Prediksi",
    ]

    page_labels = {
        "Home":                     "Home",
        "Dataset":                  "Dataset",
        "Preprocessing & Augmentasi": "Preprocessing",
        "Training":                 "Training",
        "Riwayat Training":         "Riwayat Training",
        "Evaluasi":                 "Evaluasi",
        "Prediksi":                 "Prediksi",
    }

    dataset_ok  = get_state("dataset_ready", False)
    split_ok    = get_state("split_ready", False)
    trained_any = bool(get_state("training_history", {}))
    eval_ok     = bool(get_state("eval_results", {}))

    page = st.radio("Navigasi", pages,
                    format_func=lambda x: page_labels.get(x, x),
                    label_visibility="collapsed")

    st.divider()
    st.markdown("**Status Alur**")
    if st.button("Reset Semua State", use_container_width=True):
        for k in ["dataset_ready","split_ready","aug_config","training_history",
                  "eval_results","class_names","num_classes","dataset_root",
                  "split_info","best_model"]:
            if k in st.session_state:
                del st.session_state[k]
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
        st.rerun()

    steps = [
        ("Dataset",     dataset_ok),
        ("Split",       split_ok),
        ("Preprocess",  split_ok),
        ("Training",    trained_any),
        ("Evaluasi",    eval_ok),
    ]

    status_html = '<div class="status-grid">'
    for label, done in steps:
        cls = "status-item done" if done else "status-item"
        status_html += f'<div class="{cls}"><span class="status-dot"></span>{label}</div>'
    status_html += '</div>'
    st.markdown(status_html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""
    <div style="
        background: linear-gradient(120deg, #0ea5e9 0%, #2e7d32 60%, #43a047 100%);
        border-radius: 16px;
        padding: 36px 40px 28px 40px;
        margin-bottom: 8px;
        box-shadow: 0 4px 24px rgba(14,165,233,0.15);
    ">
        <div style="font-size: 2.2rem; font-weight: 800; color: #ffffff; letter-spacing: -0.5px; line-height: 1.2;">
            Sistem Klasifikasi Rempah Indonesia
        </div>
        <div style="font-size: 1.05rem; color: #d1fae5; margin-top: 10px; font-weight: 400;">
            Berbasis <b style="color:#ffffff;">Convolutional Neural Network (CNN)</b> &amp; Transfer Learning
        </div>
        <div style="display:flex; gap: 12px; margin-top: 20px; flex-wrap: wrap;">
            <span style="background:rgba(255,255,255,0.18); color:#fff; border-radius:20px; padding:4px 14px; font-size:0.82rem; font-weight:600;">VGG16</span>
            <span style="background:rgba(255,255,255,0.18); color:#fff; border-radius:20px; padding:4px 14px; font-size:0.82rem; font-weight:600;">ResNet50</span>
            <span style="background:rgba(255,255,255,0.18); color:#fff; border-radius:20px; padding:4px 14px; font-size:0.82rem; font-weight:600;">EfficientNetB0</span>
            <span style="background:rgba(255,255,255,0.18); color:#fff; border-radius:20px; padding:4px 14px; font-size:0.82rem; font-weight:600;">MobileNetV2</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class="info-box">
        <b>Tentang Sistem</b><br>
        Sistem ini merupakan implementasi penelitian klasifikasi citra rempah Indonesia
        menggunakan metode <b>Transfer Learning CNN</b>. Model yang digunakan:
        VGG16, ResNet50, EfficientNetB0, dan MobileNetV2.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Alur Penelitian")
        alur = [
            ("1", "Upload & Eksplorasi Dataset", "Upload ZIP dataset rempah, lihat distribusi kelas"),
            ("2", "Split Dataset",               "Bagi data: Train 70% / Val 15% / Test 15%"),
            ("3", "Preprocessing & Augmentasi",  "Resize, normalisasi, dan augmentasi gambar"),
            ("4", "Training Model",              "Pilih arsitektur CNN & latih model"),
            ("5", "Riwayat Training",            "Pantau grafik accuracy & loss setiap model"),
            ("6", "Evaluasi",                    "Accuracy, Precision, Recall, F1, Confusion Matrix"),
            ("7", "Prediksi",                    "Upload foto rempah dan dapatkan prediksi"),
        ]
        for num, title, desc in alur:
            st.markdown(f"""
            <div class="flow-step">
                <div class="flow-num">{num}</div>
                <div>
                    <div class="flow-title">{title}</div>
                    <div class="flow-desc">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Spesifikasi Sistem")
        specs = {
            "Input Size": "224 x 224 piksel",
            "Jumlah Kelas": "31 kelas rempah",
            "Split Ratio": "70% / 15% / 15%",
            "Optimizer": "Adam",
            "Loss Function": "Categorical Crossentropy",
            "Fine-tuning": "2 Stage Training",
            "Regularisasi": "L2 + Dropout + BatchNorm",
            "Callbacks": "EarlyStopping + ReduceLR",
        }
        for k, v in specs.items():
            st.markdown(f"**{k}:** {v}")

        st.divider()
        st.markdown("### Arsitektur Model")
        models_info = {
            "VGG16":          "16 layer, 138M params",
            "ResNet50":       "50 layer, 25M params",
            "EfficientNetB0": "Compound scaling",
            "MobileNetV2":    "Lightweight, mobile",
        }
        for m, d in models_info.items():
            st.markdown(f"- **{m}**: {d}")

# ═══════════════════════════════════════════════════════════════════
# PAGE: DATASET
# ═══════════════════════════════════════════════════════════════════
elif page == "Dataset":
    st.title("Dataset")
    st.markdown('<span class="step-badge">Langkah 1</span>', unsafe_allow_html=True)

    st.markdown("### Upload Dataset ZIP")
    st.markdown("""
    <div class="info-box">
    Upload file ZIP dengan struktur folder: <code>nama_kelas/gambar.jpg</code><br>
    Setiap subfolder = 1 kelas rempah.
    </div>
    """, unsafe_allow_html=True)

    uploaded_zip = st.file_uploader(
        "Pilih file ZIP dataset", type=["zip"],
        help="Struktur ZIP: kelas1/img1.jpg, kelas2/img1.jpg, ..."
    )

    # Tombol reset dataset
    if get_state("dataset_ready"):
        if st.button("Reset / Ganti Dataset", type="secondary"):
            set_state("dataset_ready", False)
            set_state("split_ready", False)
            set_state("class_names", [])
            set_state("num_classes", 0)
            set_state("dataset_root", "")
            st.rerun()

    if uploaded_zip:
        if st.button("Ekstrak & Analisis Dataset", type="primary"):
            with st.spinner("Mengekstrak dataset..."):
                if os.path.exists(DATASET_DIR):
                    shutil.rmtree(DATASET_DIR)
                os.makedirs(DATASET_DIR, exist_ok=True)

                zip_path = f"{BASE_DIR}/dataset.zip"
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.getbuffer())

                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(DATASET_DIR)

                roots = []
                for root, dirs, files in os.walk(DATASET_DIR):
                    dirs[:] = [d for d in dirs if not d.startswith("__") and not d.startswith(".")]
                    imgs = [f for f in files if f.lower().endswith((".jpg",".jpeg",".png"))]
                    if imgs:
                        roots.append(root)
                if roots:
                    set_state("dataset_root", os.path.dirname(roots[0]))
                    set_state("dataset_ready", True)
                    st.success("Dataset berhasil diekstrak!")
                    st.rerun()
                else:
                    st.error("Tidak ada gambar ditemukan dalam ZIP. Periksa struktur folder.")
    else:
        if not get_state("dataset_ready"):
            st.markdown("""
            <div class="warn-box">
            Belum ada dataset. Silakan upload file ZIP dataset di atas.
            </div>
            """, unsafe_allow_html=True)

    if get_state("dataset_ready"):
        dataset_root = get_state("dataset_root", DATASET_DIR)

        # Validasi folder masih ada
        if not os.path.exists(dataset_root):
            st.error(f"Folder dataset tidak ditemukan: `{dataset_root}`\n\nSilakan upload ulang dataset.")
            set_state("dataset_ready", False)
            st.stop()

        # Fungsi scan subfolder berisi gambar
        def scan_kelas(root):
            rows, names = [], []
            for kelas in sorted(os.listdir(root)):
                kdir = os.path.join(root, kelas)
                if not os.path.isdir(kdir):
                    continue
                if kelas.startswith((".", "__", "hasil_")):
                    continue
                imgs = [f for f in os.listdir(kdir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                if imgs:
                    names.append(kelas)
                    rows.append({"Kelas": kelas, "Jumlah Gambar": len(imgs)})
            return rows, names

        data_rows, kelas_list = scan_kelas(dataset_root)

        # ZIP mungkin punya subfolder pembungkus (mis. dataset/nama_dataset/kelas/img.jpg)
        if not data_rows:
            subdirs = [
                os.path.join(dataset_root, d)
                for d in os.listdir(dataset_root)
                if os.path.isdir(os.path.join(dataset_root, d))
                and not d.startswith((".", "__"))
            ]
            for subdir in subdirs:
                data_rows, kelas_list = scan_kelas(subdir)
                if data_rows:
                    dataset_root = subdir
                    set_state("dataset_root", dataset_root)
                    break

        # Jika masih kosong, tampilkan info struktur untuk debug
        if not data_rows:
            st.error("Tidak ditemukan gambar dalam subfolder manapun.")
            st.markdown("**Struktur folder yang ditemukan:**")
            tree_lines = []
            for root_w, dirs, files in os.walk(dataset_root):
                dirs[:] = [d for d in dirs if not d.startswith((".", "__"))]
                level = root_w.replace(dataset_root, "").count(os.sep)
                if level > 3:
                    break
                indent = "  " * level
                tree_lines.append(f"{indent}{os.path.basename(root_w) or 'root'}/")
                img_files = [f for f in files if f.lower().endswith((".jpg",".jpeg",".png"))]
                if img_files:
                    tree_lines.append(f"{indent}  [{len(img_files)} gambar]")
            st.code("\n".join(tree_lines[:60]))
            st.info("Struktur ZIP yang benar: `nama_kelas/gambar.jpg` (setiap subfolder = 1 kelas)")
            st.stop()

        df_dist = pd.DataFrame(data_rows)   # kolom: "Kelas", "Jumlah Gambar"
        set_state("class_names", kelas_list)
        set_state("num_classes", len(kelas_list))

        total_gambar = int(df_dist["Jumlah Gambar"].sum())
        rata2_gambar = int(df_dist["Jumlah Gambar"].mean())

        st.divider()
        st.markdown("### Informasi Dataset")

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""<div class="metric-card"><div class="val">{len(kelas_list)}</div>
        <div class="lbl">Jumlah Kelas</div></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card"><div class="val">{total_gambar}</div>
        <div class="lbl">Total Gambar</div></div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-card"><div class="val">{rata2_gambar}</div>
        <div class="lbl">Rata-rata/Kelas</div></div>""", unsafe_allow_html=True)

        st.markdown("#### Distribusi Kelas")
        tab1, tab2 = st.tabs(["Grafik", "Tabel"])
        with tab1:
            fig, ax = plt.subplots(figsize=(14, 5))
            colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(df_dist)))
            ax.bar(df_dist["Kelas"], df_dist["Jumlah Gambar"], color=colors, edgecolor="white")
            ax.set_xticklabels(df_dist["Kelas"], rotation=75, ha="right", fontsize=9)
            ax.set_title("Distribusi Jumlah Citra per Kelas Rempah", fontsize=13, fontweight="bold")
            ax.set_xlabel("Kelas Rempah"); ax.set_ylabel("Jumlah Gambar")
            ax.grid(axis="y", alpha=0.3); plt.tight_layout()
            st.pyplot(fig); plt.close()
        with tab2:
            st.dataframe(df_dist, use_container_width=True)

        st.markdown("#### Preview Gambar per Kelas")
        num_preview = min(len(kelas_list), 8)
        cols = st.columns(num_preview)
        for i, kelas in enumerate(kelas_list[:num_preview]):
            kdir = os.path.join(dataset_root, kelas)
            imgs = [f for f in os.listdir(kdir) if f.lower().endswith((".jpg",".jpeg",".png"))]
            if imgs:
                img_path = os.path.join(kdir, random.choice(imgs))
                try:
                    img = Image.open(img_path).convert("RGB")
                    cols[i].image(img, caption=kelas, use_container_width=True)
                except:
                    cols[i].write(kelas)

        st.divider()
        st.markdown("### Split Dataset")

        col_s1, col_s2, col_s3 = st.columns(3)
        train_ratio = col_s1.slider("Train %", 50, 80, 70, 5)
        val_ratio   = col_s2.slider("Validasi %", 5, 30, 15, 5)
        test_ratio  = 100 - train_ratio - val_ratio
        col_s3.metric("Test %", f"{test_ratio}%")

        if test_ratio <= 0:
            st.error("Persentase Test harus > 0. Kurangi Train atau Validasi.")
        else:
            st.info(f"**Split:** Train {train_ratio}% / Val {val_ratio}% / Test {test_ratio}%")

            if st.button("Lakukan Split Dataset", type="primary"):
                with st.spinner("Memproses split dataset..."):
                    all_data = []
                    for kelas in kelas_list:
                        kdir = os.path.join(dataset_root, kelas)
                        for f in os.listdir(kdir):
                            if f.lower().endswith((".jpg",".jpeg",".png")):
                                all_data.append({"filepath": os.path.join(kdir, f), "label": kelas})

                    df_all = pd.DataFrame(all_data)
                    val_test_ratio = (val_ratio + test_ratio) / 100
                    test_of_temp   = test_ratio / (val_ratio + test_ratio)

                    train_df, temp_df = train_test_split(
                        df_all, test_size=val_test_ratio,
                        stratify=df_all["label"], random_state=42
                    )
                    val_df, test_df = train_test_split(
                        temp_df, test_size=test_of_temp,
                        stratify=temp_df["label"], random_state=42
                    )

                    # Simpan CSV dengan encoding utf-8 (tanpa BOM)
                    train_df.to_csv(f"{LAPORAN_DIR}/train.csv", index=False, encoding="utf-8")
                    val_df.to_csv(f"{LAPORAN_DIR}/validation.csv", index=False, encoding="utf-8")
                    test_df.to_csv(f"{LAPORAN_DIR}/test.csv", index=False, encoding="utf-8")

                    set_state("split_ready", True)
                    set_state("split_info", {
                        "total": len(df_all),
                        "train": len(train_df),
                        "val":   len(val_df),
                        "test":  len(test_df),
                    })

                    st.success("Split berhasil!")

        if get_state("split_ready"):
            si = get_state("split_info", {})
            if si:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total", si.get("total","—"))
                c2.metric("Train", si.get("train","—"))
                c3.metric("Validasi", si.get("val","—"))
                c4.metric("Test", si.get("test","—"))

# ═══════════════════════════════════════════════════════════════════
# PAGE: PREPROCESSING & AUGMENTASI
# ═══════════════════════════════════════════════════════════════════
elif page == "Preprocessing & Augmentasi":
    st.title("Preprocessing & Augmentasi")
    st.markdown('<span class="step-badge">Langkah 3</span>', unsafe_allow_html=True)

    if not get_state("split_ready"):
        st.markdown("""<div class="warn-box">Harap selesaikan <b>Upload Dataset</b> dan <b>Split Dataset</b> terlebih dahulu.</div>""",
                    unsafe_allow_html=True)
        st.stop()

    st.markdown("### Preprocessing (Otomatis)")
    c1, c2, c3 = st.columns(3)
    c1.success("**Resize** ke 224x224 px")
    c2.success("**Normalisasi** ke [0,1]")
    c3.success("**Label Encoding** One-hot")

    with st.expander("Detail Preprocessing", expanded=False):
        st.markdown("""
        - **Resize 224x224**: Semua gambar diubah ke ukuran seragam sesuai input CNN
        - **Normalisasi**: Nilai piksel dibagi 255 (atau menggunakan preprocess_input spesifik model)
        - **Label Encoding**: Label string dikonversi ke one-hot vector sesuai jumlah kelas
        - **Channel**: RGB (3 channel)
        """)

    st.divider()

    st.markdown("### Konfigurasi Augmentasi")

    aug_mode = st.radio(
        "Mode Augmentasi",
        ["Tanpa Augmentasi", "Augmentasi Standar", "Augmentasi Kustom"],
        horizontal=True
    )

    aug_config = {}

    if aug_mode == "Tanpa Augmentasi":
        st.info("Tidak ada augmentasi diterapkan. Hanya preprocessing standar.")
        aug_config = {"mode": "none"}

    elif aug_mode == "Augmentasi Standar":
        st.success("Menggunakan konfigurasi augmentasi dari penelitian:")
        cfg_cols = st.columns(3)
        cfg_cols[0].markdown("- Horizontal Flip\n- Rotation 20 derajat")
        cfg_cols[1].markdown("- Width Shift 15%\n- Height Shift 15%")
        cfg_cols[2].markdown("- Zoom 15%\n- Brightness 0.8 s.d. 1.2")
        aug_config = {
            "mode": "standard",
            "horizontal_flip": True,
            "rotation_range": 20,
            "width_shift_range": 0.15,
            "height_shift_range": 0.15,
            "zoom_range": 0.15,
            "brightness_min": 0.8,
            "brightness_max": 1.2,
        }

    else:
        st.markdown("**Pilih teknik augmentasi:**")
        col_a, col_b = st.columns(2)
        with col_a:
            h_flip   = st.checkbox("Horizontal Flip", value=True)
            rotation = st.slider("Rotation Range (derajat)", 0, 45, 20)
            zoom     = st.slider("Zoom Range", 0.0, 0.5, 0.15)
        with col_b:
            width_s  = st.slider("Width Shift Range", 0.0, 0.4, 0.15)
            height_s = st.slider("Height Shift Range", 0.0, 0.4, 0.15)
            br_min   = st.slider("Brightness Min", 0.5, 1.0, 0.8)
            br_max   = st.slider("Brightness Max", 1.0, 1.5, 1.2)

        aug_config = {
            "mode": "custom",
            "horizontal_flip": h_flip,
            "rotation_range": rotation,
            "width_shift_range": width_s,
            "height_shift_range": height_s,
            "zoom_range": zoom,
            "brightness_min": br_min,
            "brightness_max": br_max,
        }

    if st.button("Simpan Konfigurasi Augmentasi", type="primary"):
        set_state("aug_config", aug_config)
        st.success("Konfigurasi augmentasi disimpan!")

    st.divider()
    st.markdown("### Preview Hasil Augmentasi")

    dataset_root = get_state("dataset_root", DATASET_DIR)
    kelas_list   = get_state("class_names", [])

    if kelas_list:
        sel_kelas = st.selectbox("Pilih kelas untuk preview:", kelas_list)

        if st.button("Generate Preview Augmentasi"):
            kdir = os.path.join(dataset_root, sel_kelas)
            imgs = [f for f in os.listdir(kdir) if f.lower().endswith((".jpg",".jpeg",".png"))]

            if imgs:
                try:
                    from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img

                    img_path = os.path.join(kdir, random.choice(imgs))
                    img_orig = Image.open(img_path).convert("RGB").resize((224, 224))
                    img_arr  = img_to_array(img_orig)
                    img_arr  = img_arr.reshape((1,) + img_arr.shape)

                    cfg = get_state("aug_config", aug_config)
                    if cfg.get("mode") == "none":
                        datagen = ImageDataGenerator()
                    else:
                        datagen = ImageDataGenerator(
                            rotation_range=cfg.get("rotation_range", 20),
                            width_shift_range=cfg.get("width_shift_range", 0.15),
                            height_shift_range=cfg.get("height_shift_range", 0.15),
                            zoom_range=cfg.get("zoom_range", 0.15),
                            horizontal_flip=cfg.get("horizontal_flip", True),
                            brightness_range=[cfg.get("brightness_min", 0.8), cfg.get("brightness_max", 1.2)],
                            fill_mode="nearest"
                        )

                    aug_imgs = [img_orig]
                    labels   = ["Asli"]
                    aug_names = ["Flip", "Rotate", "Zoom", "Shift", "Brightness", "Combined"]

                    i = 0
                    for batch in datagen.flow(img_arr, batch_size=1, seed=42):
                        aug_imgs.append(array_to_img(batch[0]))
                        labels.append(aug_names[i] if i < len(aug_names) else f"Aug {i+1}")
                        i += 1
                        if i >= 6:
                            break

                    fig, axes = plt.subplots(1, len(aug_imgs), figsize=(3 * len(aug_imgs), 3.5))
                    if len(aug_imgs) == 1:
                        axes = [axes]
                    for ax, im, lbl in zip(axes, aug_imgs, labels):
                        ax.imshow(im)
                        ax.set_title(lbl, fontsize=9, fontweight="bold")
                        ax.axis("off")
                    plt.suptitle(f"Preview Augmentasi — {sel_kelas}", fontsize=12, fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                except ImportError:
                    st.warning("TensorFlow tidak tersedia. Preview augmentasi menggunakan PIL.")
                    img_path = os.path.join(kdir, random.choice(imgs))
                    img_pil  = Image.open(img_path).convert("RGB").resize((224, 224))
                    previews = [("Asli", img_pil)]
                    try:
                        previews.append(("H-Flip", img_pil.transpose(Image.FLIP_LEFT_RIGHT)))
                        from PIL import ImageEnhance
                        previews.append(("Brightness+", ImageEnhance.Brightness(img_pil).enhance(1.2)))
                        previews.append(("Brightness-", ImageEnhance.Brightness(img_pil).enhance(0.8)))
                    except:
                        pass
                    cols_p = st.columns(len(previews))
                    for col, (lbl, im) in zip(cols_p, previews):
                        col.image(im, caption=lbl, use_container_width=True)
    else:
        st.info("Dataset belum tersedia. Upload dataset terlebih dahulu.")

# ═══════════════════════════════════════════════════════════════════
# PAGE: TRAINING
# ═══════════════════════════════════════════════════════════════════
elif page == "Training":
    st.title("Training Model")
    st.markdown('<span class="step-badge">Langkah 4</span>', unsafe_allow_html=True)

    if not get_state("split_ready"):
        st.markdown("""<div class="warn-box">Harap selesaikan Dataset & Split terlebih dahulu.</div>""",
                    unsafe_allow_html=True)
        st.stop()

    fast_mode = get_state("fast_mode", False)

    st.markdown("""
    <div style="background:linear-gradient(120deg,#0ea5e9,#2e7d32);border-radius:12px;
                padding:16px 20px;margin-bottom:16px;">
        <div style="color:#fff;font-weight:700;font-size:1rem;">Mode Cepat</div>
        <div style="color:#d1fae5;font-size:0.82rem;margin-top:2px;">
            MobileNetV2 · Batch 32 · Epoch 2+1 · EarlyStopping 2 · Tanpa Augmentasi · 128x128
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_fm1, col_fm2 = st.columns([1, 3])
    with col_fm1:
        if st.button("Aktifkan Mode Cepat" if not fast_mode else "Nonaktifkan Mode Cepat",
                     type="primary" if not fast_mode else "secondary",
                     use_container_width=True):
            set_state("fast_mode", not fast_mode)
            st.rerun()

    if fast_mode:
        st.success("Mode Cepat aktif — cocok untuk laptop RAM 8GB tanpa GPU.")

    st.divider()
    st.markdown("### Pilih Arsitektur Model")

    model_options = {
        "VGG16": {"params": "138M", "depth": "16 layer", "desc": "Deep & powerful"},
        "ResNet50": {"params": "25M", "depth": "50 layer", "desc": "Skip connections"},
        "EfficientNetB0": {"params": "5.3M", "depth": "~82 layer", "desc": "Efisien & akurat"},
        "MobileNetV2": {"params": "3.4M", "depth": "~53 layer", "desc": "Ringan untuk demo"},
    }

    default_models = {"MobileNetV2"} if fast_mode else {"VGG16"}
    sel_models = []
    cols_m = st.columns(4)

    for i, (m, info) in enumerate(model_options.items()):
        with cols_m[i]:
            checked = st.checkbox(
                f"**{m}**",
                value=(m in default_models),
                key=f"chk_{m}",
                disabled=fast_mode
            )
            if fast_mode:
                sel_models = ["MobileNetV2"]
            elif checked:
                sel_models.append(m)
            st.caption(f"{info['params']} · {info['depth']}")
            st.caption(info["desc"])

    sel_models = list(dict.fromkeys(sel_models))

    st.divider()
    st.markdown("### Hyperparameter")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)

    if fast_mode:
        epochs1 = 1
        epochs2 = 1
        batch_size = 32
        img_size = 128
        lr1        = col_p4.select_slider(
            "Learning Rate Stage 1",
            options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            value=1e-3,
            format_func=lambda x: f"{x:.0e}"
        )
    else:
        epochs1    = col_p1.number_input("Epoch Stage 1", 1, 30, 10)
        epochs2    = col_p2.number_input("Epoch Stage 2", 1, 30, 13)
        batch_size = col_p3.selectbox("Batch Size", [8, 16, 32, 64], index=2)
        img_size   = 224
        lr1        = col_p4.select_slider(
            "Learning Rate Stage 1",
            options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            value=5e-4,
            format_func=lambda x: f"{x:.0e}"
        )

    with st.expander("Advanced Settings"):
        col_a1, col_a2 = st.columns(2)
        l2_reg       = col_a1.number_input("L2 Regularization", 1e-5, 1e-2, 1e-4, format="%.5f")
        dropout1     = col_a1.slider("Dropout Layer 1", 0.1, 0.7, 0.3 if fast_mode else 0.4)
        dropout2     = col_a2.slider("Dropout Layer 2", 0.0, 0.5, 0.1 if fast_mode else 0.2)
        patience_es  = col_a2.number_input("EarlyStopping Patience", 1, 10, 2 if fast_mode else 4)
        label_smooth = col_a1.slider("Label Smoothing", 0.0, 0.2, 0.0 if fast_mode else 0.05)

    st.divider()
    if fast_mode:
        st.markdown("""
        <div class="info-box">
        <b>Estimasi Mode Cepat Laptop RAM 8GB:</b><br>
        MobileNetV2, gambar 128x128, Batch 32, Epoch 2+1, tanpa augmentasi.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warn-box">
        Mode normal lebih berat. Untuk laptop tanpa GPU, gunakan Mode Cepat.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    mode_training = st.radio(
        "Mode Training",
        ["Demo Cepat", "Training Full"],
        horizontal=True
    )

    if sel_models and st.button("Mulai Training", type="primary"):
        try:
            import tensorflow as tf
            from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
            from tensorflow.keras.applications.efficientnet import EfficientNetB0
            from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
            from tensorflow.keras.applications.resnet50 import preprocess_input as res_pre
            from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_pre
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
            from tensorflow.keras.models import Model
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.regularizers import l2 as l2_reg_fn
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

            PREPROCESS = {"VGG16": vgg_pre, "ResNet50": res_pre,
                          "EfficientNetB0": eff_pre, "MobileNetV2": mob_pre}
            ARCH = {"VGG16": VGG16, "ResNet50": ResNet50,
                    "EfficientNetB0": EfficientNetB0, "MobileNetV2": MobileNetV2}
            FINETUNE_LAYERS = {"VGG16": -12, "ResNet50": -20,
                               "EfficientNetB0": -30, "MobileNetV2": -40}

            # ── Baca & bersihkan CSV ─────────────────────────────────────────
            train_df = pd.read_csv(f"{LAPORAN_DIR}/train.csv", encoding="utf-8-sig")
            val_df   = pd.read_csv(f"{LAPORAN_DIR}/validation.csv", encoding="utf-8-sig")

            train_df = clean_csv_dataframe(train_df)
            val_df   = clean_csv_dataframe(val_df)

            # Validasi kolom wajib tersedia sebelum lanjut
            for col in ["filepath", "label"]:
                if col not in train_df.columns:
                    st.error(
                        f"Kolom '{col}' tidak ditemukan setelah cleaning.\n"
                        f"Kolom train.csv: {train_df.columns.tolist()}"
                    )
                    st.stop()

            # Debug info
            with st.expander("Debug Info CSV", expanded=False):
                st.write("Kolom train:", train_df.columns.tolist())
                st.write("Kolom val:", val_df.columns.tolist())
                st.write("Contoh label train:", train_df['label'].unique()[:5].tolist())
                st.write("Contoh filepath:", train_df['filepath'].iloc[0])
                st.write("Jumlah train:", len(train_df), "| Jumlah val:", len(val_df))
            # ────────────────────────────────────────────────────────────────

            num_classes = train_df["label"].nunique()

            if mode_training == "Demo Cepat":
                # Gunakan concat+sample per kelas agar kolom tidak hilang
                # (pandas groupby().apply() pada versi tertentu bisa drop kolom groupby)
                sampled_train = []
                for lbl, grp in train_df.groupby("label"):
                    sampled_train.append(grp.sample(min(len(grp), 20), random_state=42))
                train_df = pd.concat(sampled_train, ignore_index=True)

                sampled_val = []
                for lbl, grp in val_df.groupby("label"):
                    sampled_val.append(grp.sample(min(len(grp), 5), random_state=42))
                val_df = pd.concat(sampled_val, ignore_index=True)

                st.warning(
                    f"Mode Demo Cepat aktif: "
                    f"{len(train_df)} data train dan {len(val_df)} data validasi digunakan."
                )

            if get_state("fast_mode", False):
                aug_cfg = {"mode": "none"}
            else:
                aug_cfg = get_state("aug_config", {"mode": "standard"})

            training_history = get_state("training_history", {})

            for model_name in sel_models:
                st.markdown(f"---\n#### Training **{model_name}**")
                prog = st.progress(0, text=f"Memulai {model_name}...")

                preprocess_fn = PREPROCESS[model_name]

                if aug_cfg.get("mode") == "none":
                    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
                else:
                    train_datagen = ImageDataGenerator(
                        preprocessing_function=preprocess_fn,
                        rotation_range=aug_cfg.get("rotation_range", 20),
                        width_shift_range=aug_cfg.get("width_shift_range", 0.15),
                        height_shift_range=aug_cfg.get("height_shift_range", 0.15),
                        zoom_range=aug_cfg.get("zoom_range", 0.15),
                        horizontal_flip=aug_cfg.get("horizontal_flip", True),
                        brightness_range=[aug_cfg.get("brightness_min", 0.8),
                                          aug_cfg.get("brightness_max", 1.2)],
                    )

                val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

                # ── Hard-check kolom sebelum flow_from_dataframe ──────────
                if "label" not in train_df.columns or "filepath" not in train_df.columns:
                    st.error(
                        f"**train_df kehilangan kolom sebelum flow_from_dataframe!**\n\n"
                        f"Kolom yang ada: `{train_df.columns.tolist()}`\n\n"
                        f"Cek apakah file train.csv sudah benar, lalu lakukan Split ulang."
                    )
                    st.stop()
                # ──────────────────────────────────────────────────────────

                train_gen = train_datagen.flow_from_dataframe(
                    train_df,
                    x_col="filepath",
                    y_col="label",
                    target_size=(img_size, img_size),
                    batch_size=batch_size,
                    class_mode="categorical",
                    shuffle=True,
                    seed=42
                )

                val_gen = val_datagen.flow_from_dataframe(
                    val_df,
                    x_col="filepath",
                    y_col="label",
                    target_size=(img_size, img_size),
                    batch_size=batch_size,
                    class_mode="categorical",
                    shuffle=False
                )

                base = ARCH[model_name](
                    weights="imagenet",
                    include_top=False,
                    input_shape=(img_size, img_size, 3)
                )

                base.trainable = False
                x = base.output
                x = GlobalAveragePooling2D()(x)
                x = Dense(256, activation="relu", kernel_regularizer=l2_reg_fn(l2_reg))(x)
                x = BatchNormalization()(x)
                x = Dropout(dropout1)(x)
                x = Dense(128, activation="relu", kernel_regularizer=l2_reg_fn(l2_reg))(x)
                x = BatchNormalization()(x)
                x = Dropout(dropout2)(x)
                out = Dense(num_classes, activation="softmax")(x)
                mdl = Model(inputs=base.input, outputs=out)

                callbacks = [
                    EarlyStopping(monitor="val_loss", patience=patience_es, restore_best_weights=True),
                    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=0)
                ]

                prog.progress(10, text=f"{model_name} — Stage 1: Head Training...")
                mdl.compile(
                    optimizer=Adam(learning_rate=lr1),
                    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smooth),
                    metrics=["accuracy"]
                )

                t0 = time.time()
                h1 = mdl.fit(train_gen, validation_data=val_gen,
                             epochs=epochs1, callbacks=callbacks, verbose=0)
                t1 = (time.time() - t0) / 60

                prog.progress(55, text=f"{model_name} — Stage 2: Fine-tuning...")
                base.trainable = True
                ft = FINETUNE_LAYERS[model_name]
                for layer in base.layers[:ft]:
                    layer.trainable = False

                mdl.compile(
                    optimizer=Adam(learning_rate=lr1 / 10),
                    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smooth),
                    metrics=["accuracy"]
                )

                t2s = time.time()
                h2 = mdl.fit(train_gen, validation_data=val_gen,
                             epochs=epochs2, callbacks=callbacks, verbose=0)
                t2 = (time.time() - t2s) / 60

                prog.progress(85, text=f"{model_name} — Menyimpan model...")
                model_path = f"{MODEL_DIR}/{model_name.lower()}_model.h5"
                mdl.save(model_path)

                acc = h1.history["accuracy"] + h2.history["accuracy"]
                val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
                loss = h1.history["loss"] + h2.history["loss"]
                val_loss = h1.history["val_loss"] + h2.history["val_loss"]

                training_history[model_name] = {
                    "acc": acc,
                    "val_acc": val_acc,
                    "loss": loss,
                    "val_loss": val_loss,
                    "time_stage1": round(t1, 2),
                    "time_stage2": round(t2, 2),
                    "epochs_total": len(acc),
                    "class_indices": train_gen.class_indices,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M"),
                    "img_size": img_size,
                    "batch_size": batch_size,
                    "fast_mode": fast_mode,
                }

                prog.progress(100, text=f"{model_name} selesai!")
                st.success(f"**{model_name}** selesai — Stage1: {t1:.1f} mnt | Stage2: {t2:.1f} mnt")

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.plot(acc, label="Train")
                ax1.plot(val_acc, label="Val")
                ax1.set_title(f"{model_name} — Accuracy")
                ax1.legend()
                ax1.grid(alpha=.3)

                ax2.plot(loss, label="Train")
                ax2.plot(val_loss, label="Val")
                ax2.set_title(f"{model_name} — Loss")
                ax2.legend()
                ax2.grid(alpha=.3)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            set_state("training_history", training_history)
            st.balloons()
            st.success("Training selesai! Lanjut ke halaman Evaluasi.")

        except ImportError:
            st.error("TensorFlow tidak terinstall. Jalankan: pip install tensorflow")
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════
# PAGE: RIWAYAT TRAINING
# ═══════════════════════════════════════════════════════════════════
elif page == "Riwayat Training":
    st.title("Riwayat Training")

    history = get_state("training_history", {})

    if not history:
        st.markdown("""<div class="warn-box">Belum ada model yang dilatih.</div>""",
                    unsafe_allow_html=True)
        st.stop()

    st.markdown(f"**Total model terlatih:** {len(history)}")

    for model_name, h in history.items():
        with st.expander(f"{model_name} — Terlatih: {h.get('timestamp','—')}", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Epoch Total",    h.get("epochs_total", "—"))
            c2.metric("Best Train Acc", f"{max(h['acc']):.4f}")
            c3.metric("Best Val Acc",   f"{max(h['val_acc']):.4f}")
            c4.metric("Waktu (mnt)",    f"{h.get('time_stage1',0)+h.get('time_stage2',0):.1f}")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            epochs = range(1, len(h["acc"]) + 1)
            ax1.plot(epochs, h["acc"], "b-o", ms=3, label="Train Acc")
            ax1.plot(epochs, h["val_acc"], "g-o", ms=3, label="Val Acc")
            ax1.set_title(f"{model_name} — Accuracy Curve")
            ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
            ax1.legend(); ax1.grid(alpha=.3)

            ax2.plot(epochs, h["loss"], "r-o", ms=3, label="Train Loss")
            ax2.plot(epochs, h["val_loss"], "o-", color="orange", ms=3, label="Val Loss")
            ax2.set_title(f"{model_name} — Loss Curve")
            ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
            ax2.legend(); ax2.grid(alpha=.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    if len(history) > 1:
        st.divider()
        st.markdown("### Perbandingan Semua Model")
        comp_data = []
        for m, h in history.items():
            comp_data.append({
                "Model":          m,
                "Best Train Acc": f"{max(h['acc']):.4f}",
                "Best Val Acc":   f"{max(h['val_acc']):.4f}",
                "Final Train Acc":f"{h['acc'][-1]:.4f}",
                "Final Val Acc":  f"{h['val_acc'][-1]:.4f}",
                "Total Epoch":    h["epochs_total"],
                "Waktu (mnt)":    round(h.get("time_stage1",0)+h.get("time_stage2",0),1),
            })
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE: EVALUASI
# ═══════════════════════════════════════════════════════════════════
elif page == "Evaluasi":
    st.title("Evaluasi Model")
    st.markdown('<span class="step-badge">Langkah 6</span>', unsafe_allow_html=True)

    history = get_state("training_history", {})

    if not history:
        st.markdown("""<div class="warn-box">Belum ada model terlatih. Lakukan Training terlebih dahulu.</div>""",
                    unsafe_allow_html=True)
        st.stop()

    if not os.path.exists(f"{LAPORAN_DIR}/test.csv"):
        st.warning("File test.csv tidak ditemukan. Lakukan Split Dataset.")
        st.stop()

    if st.button("Jalankan Evaluasi Semua Model", type="primary"):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
            from tensorflow.keras.applications.resnet50 import preprocess_input as res_pre
            from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_pre
            from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                         f1_score, confusion_matrix, classification_report)

            PREPROCESS = {"vgg16": vgg_pre, "resnet50": res_pre,
                          "efficientnetb0": eff_pre, "mobilenetv2": mob_pre}

            # ── Baca & bersihkan test CSV ────────────────────────────────────
            test_df = pd.read_csv(f"{LAPORAN_DIR}/test.csv", encoding="utf-8-sig")
            test_df = clean_csv_dataframe(test_df)

            # Validasi kolom wajib tersedia sebelum lanjut
            for col in ["filepath", "label"]:
                if col not in test_df.columns:
                    st.error(
                        f"Kolom '{col}' tidak ditemukan setelah cleaning.\n"
                        f"Kolom test.csv: {test_df.columns.tolist()}"
                    )
                    st.stop()
            # ────────────────────────────────────────────────────────────────

            eval_results = {}

            for model_name in history.keys():
                model_key  = model_name.lower()
                model_path = f"{MODEL_DIR}/{model_key}_model.h5"

                if not os.path.exists(model_path):
                    st.warning(f"File model {model_path} tidak ditemukan, skip.")
                    continue

                st.info(f"Evaluasi {model_name}...")
                pre_fn = PREPROCESS.get(model_key, vgg_pre)
                test_datagen = ImageDataGenerator(preprocessing_function=pre_fn)
                img_size = history[model_name].get("img_size", 224)

                test_gen = test_datagen.flow_from_dataframe(
                    test_df, x_col="filepath", y_col="label",
                    target_size=(img_size, img_size), batch_size=32,
                    class_mode="categorical", shuffle=False)

                mdl = load_model(model_path, compile=False)
                t0 = time.time()
                y_prob = mdl.predict(test_gen, verbose=0)
                inf_ms = ((time.time()-t0)/len(test_df))*1000

                y_pred      = np.argmax(y_prob, axis=1)
                y_true      = test_gen.classes
                class_names = list(test_gen.class_indices.keys())

                acc  = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
                rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
                f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
                cm   = confusion_matrix(y_true, y_pred)
                rep  = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

                eval_results[model_name] = {
                    "accuracy":  round(acc,  4),
                    "precision": round(prec, 4),
                    "recall":    round(rec,  4),
                    "f1":        round(f1,   4),
                    "inf_ms":    round(inf_ms, 2),
                    "cm":        cm.tolist(),
                    "class_names": class_names,
                    "report":    rep,
                }

                with open(f"{LAPORAN_DIR}/classification_report_{model_key}.txt", "w") as f:
                    f.write(rep)

            set_state("eval_results", eval_results)
            st.success("Evaluasi selesai!")

        except ImportError:
            st.error("TensorFlow tidak tersedia.")
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback; st.code(traceback.format_exc())

    eval_results = get_state("eval_results", {})
    if eval_results:
        st.divider()
        st.markdown("### Hasil Evaluasi")

        rows = []
        for m, r in eval_results.items():
            rows.append({
                "Model":          m,
                "Accuracy":       r["accuracy"],
                "Precision":      r["precision"],
                "Recall":         r["recall"],
                "F1-Score":       r["f1"],
                "Inf. Time (ms)": r["inf_ms"],
            })
        df_eval  = pd.DataFrame(rows)
        best_idx = df_eval["F1-Score"].idxmax()

        best = df_eval.iloc[best_idx]
        st.markdown(f"#### Model Terbaik: **{best['Model']}**")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.markdown(f"""<div class="metric-card"><div class="val">{best['Accuracy']:.4f}</div>
        <div class="lbl">Accuracy</div></div>""", unsafe_allow_html=True)
        mc2.markdown(f"""<div class="metric-card"><div class="val">{best['Precision']:.4f}</div>
        <div class="lbl">Precision</div></div>""", unsafe_allow_html=True)
        mc3.markdown(f"""<div class="metric-card"><div class="val">{best['Recall']:.4f}</div>
        <div class="lbl">Recall</div></div>""", unsafe_allow_html=True)
        mc4.markdown(f"""<div class="metric-card"><div class="val">{best['F1-Score']:.4f}</div>
        <div class="lbl">F1-Score</div></div>""", unsafe_allow_html=True)

        st.markdown("#### Perbandingan Semua Model")
        st.dataframe(df_eval.style.highlight_max(
            subset=["Accuracy","Precision","Recall","F1-Score"], color="#c8e6c9"),
            use_container_width=True)

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        metrics = ["Accuracy","Precision","Recall","F1-Score"]
        colors  = ["#2e7d32","#1565c0","#e65100","#6a1b9a"]
        for ax, metric, color in zip(axes, metrics, colors):
            ax.bar(df_eval["Model"], df_eval[metric], color=color, alpha=.8)
            ax.set_title(metric, fontweight="bold"); ax.set_ylim(0, 1)
            ax.set_xticklabels(df_eval["Model"], rotation=30, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.divider()
        sel_eval = st.selectbox("Lihat detail model:", list(eval_results.keys()))
        r = eval_results[sel_eval]

        tab_cm, tab_rep = st.tabs(["Confusion Matrix", "Classification Report"])

        with tab_cm:
            cm = np.array(r["cm"])
            cn = r["class_names"]
            fig_size = max(10, len(cn) * 0.5)
            fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
            im = ax.imshow(cm, interpolation="nearest", cmap="Greens")
            plt.colorbar(im, ax=ax)
            tick_marks = np.arange(len(cn))
            ax.set_xticks(tick_marks); ax.set_xticklabels(cn, rotation=90, fontsize=7)
            ax.set_yticks(tick_marks); ax.set_yticklabels(cn, fontsize=7)
            thresh = cm.max() / 2
            for i in range(len(cn)):
                for j in range(len(cn)):
                    ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                            color="white" if cm[i,j] > thresh else "black", fontsize=6)
            ax.set_title(f"Confusion Matrix — {sel_eval}", fontweight="bold")
            ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with tab_rep:
            st.code(r["report"], language="text")

        best_name = df_eval.iloc[best_idx]["Model"]
        set_state("best_model", best_name)
        st.markdown(f"""
        <div class="info-box">
        <b>Model Terbaik berdasarkan F1-Score: {best_name}</b><br>
        Model ini direkomendasikan untuk digunakan pada tahap Prediksi.
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE: PREDIKSI
# ═══════════════════════════════════════════════════════════════════
elif page == "Prediksi":
    st.title("Prediksi Gambar Rempah")
    st.markdown('<span class="step-badge">Langkah 7</span>', unsafe_allow_html=True)

    history = get_state("training_history", {})
    if not history:
        st.markdown("""<div class="warn-box">Belum ada model terlatih.</div>""",
                    unsafe_allow_html=True)
        st.stop()

    best_model = get_state("best_model", list(history.keys())[0])
    available_models = [m for m in history.keys()
                        if os.path.exists(f"{MODEL_DIR}/{m.lower()}_model.h5")]

    if not available_models:
        st.warning("File model (.h5) tidak ditemukan. Pastikan training selesai dan model tersimpan.")
        st.stop()

    col_sel, col_info = st.columns([2, 1])
    with col_sel:
        sel_model = st.selectbox(
            "Pilih Model untuk Prediksi",
            available_models,
            index=available_models.index(best_model) if best_model in available_models else 0
        )
        if sel_model == best_model:
            st.success(f"**{sel_model}** — Model Terbaik (direkomendasikan)")

    with col_info:
        eval_res = get_state("eval_results", {})
        if sel_model in eval_res:
            r = eval_res[sel_model]
            st.metric("Accuracy",  r["accuracy"])
            st.metric("F1-Score", r["f1"])

    st.divider()

    st.markdown("### Upload Gambar")
    st.markdown("""
    <div class="info-box">
    Upload foto rempah dari kamera HP, unduhan internet, atau dataset baru.
    Format: JPG, JPEG, PNG
    </div>
    """, unsafe_allow_html=True)

    uploaded_imgs = st.file_uploader(
        "Upload gambar rempah (bisa lebih dari 1)",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True
    )

    if uploaded_imgs and st.button("Prediksi Sekarang", type="primary"):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
            from tensorflow.keras.applications.resnet50 import preprocess_input as res_pre
            from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_pre
            from tensorflow.keras.preprocessing.image import img_to_array

            PREPROCESS = {"vgg16": vgg_pre, "resnet50": res_pre,
                          "efficientnetb0": eff_pre, "mobilenetv2": mob_pre}

            model_key  = sel_model.lower()
            model_path = f"{MODEL_DIR}/{model_key}_model.h5"

            with st.spinner(f"Memuat model {sel_model}..."):
                mdl = load_model(model_path, compile=False)

            h = history[sel_model]
            class_indices = h.get("class_indices", {})
            if class_indices:
                class_names = [k for k, _ in sorted(class_indices.items(), key=lambda x: x[1])]
            else:
                class_names = get_state("class_names", [f"Kelas_{i}" for i in range(31)])

            pre_fn = PREPROCESS.get(model_key, vgg_pre)

            st.markdown("---")
            for img_file in uploaded_imgs:
                st.markdown(f"#### {img_file.name}")
                img_pil = Image.open(img_file).convert("RGB")

                img_size = h.get("img_size", 224)
                img_res = img_pil.resize((img_size, img_size))

                img_arr = img_to_array(img_res)
                img_arr = pre_fn(img_arr)
                img_arr = np.expand_dims(img_arr, axis=0)

                t0 = time.time()
                prob = mdl.predict(img_arr, verbose=0)[0]
                inf_ms = (time.time() - t0) * 1000

                top5_idx  = np.argsort(prob)[::-1][:5]
                top1_lbl  = class_names[top5_idx[0]] if top5_idx[0] < len(class_names) else "?"
                top1_conf = prob[top5_idx[0]] * 100

                col_img, col_res = st.columns([1, 2])

                with col_img:
                    st.image(img_pil, caption="Gambar Input", use_container_width=True)

                with col_res:
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,#1b5e20,#43a047);
                                border-radius:12px;padding:16px;color:white;text-align:center;margin-bottom:12px">
                        <div style="font-size:1.8rem;font-weight:700">{top1_lbl}</div>
                        <div style="font-size:1.2rem">{top1_conf:.2f}% confidence</div>
                        <div style="font-size:.8rem;opacity:.8">Inference: {inf_ms:.1f} ms</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("**Top-5 Prediksi:**")
                    medals = ["1.", "2.", "3.", "4.", "5."]
                    for rank, idx in enumerate(top5_idx):
                        lbl  = class_names[idx] if idx < len(class_names) else f"Kelas {idx}"
                        conf = prob[idx] * 100
                        bar_w = int(conf)
                        st.markdown(f"""
                        <div style="margin:4px 0">
                            <div style="display:flex;justify-content:space-between;margin-bottom:2px">
                                <span>{medals[rank]} {lbl}</span><span><b>{conf:.2f}%</b></span>
                            </div>
                            <div style="background:#e0e0e0;border-radius:4px;height:8px">
                                <div style="background:#2e7d32;width:{bar_w}%;height:8px;border-radius:4px"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("---")

        except ImportError:
            st.error("TensorFlow tidak terinstall.")
        except Exception as e:
            st.error(f"Error prediksi: {e}")
            import traceback; st.code(traceback.format_exc())

    elif uploaded_imgs:
        st.markdown("### Preview Gambar yang Diupload")
        preview_cols = st.columns(min(len(uploaded_imgs), 4))
        for i, img_file in enumerate(uploaded_imgs):
            img = Image.open(img_file).convert("RGB")
            preview_cols[i % 4].image(img, caption=img_file.name, use_container_width=True)