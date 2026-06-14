# 🌿 Klasifikasi Rempah Indonesia — Streamlit App
Aplikasi web berbasis Streamlit untuk penelitian klasifikasi citra rempah Indonesia menggunakan CNN Transfer Learning.

## 🚀 Cara Menjalankan

### 1. Install dependensi
```bash
pip install -r requirements.txt
```

### 2. Jalankan aplikasi
```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser: `http://localhost:8501`

---

## 📋 Alur Penggunaan

```
🏠 Home → 📂 Dataset → 🖼️ Preprocessing & Augmentasi → 🧠 Training → 📝 Riwayat → 📊 Evaluasi → 🔍 Prediksi
```

### 1. 📂 Dataset
- Upload ZIP dataset rempah (struktur: `nama_kelas/gambar.jpg`)
- Lihat distribusi kelas, jumlah gambar, preview gambar
- Lakukan split: Train 70% / Val 15% / Test 15% (bisa dikustom)

### 2. 🖼️ Preprocessing & Augmentasi
- Preprocessing otomatis: Resize 224×224, Normalisasi, Label Encoding
- Pilih mode augmentasi:
  - Tanpa Augmentasi
  - Augmentasi Standar (sesuai penelitian)
  - Augmentasi Kustom (atur sendiri setiap parameter)
- Preview hasil augmentasi per kelas

### 3. 🧠 Training
- Pilih 1 atau lebih model: VGG16, ResNet50, EfficientNetB0, MobileNetV2
- Atur hyperparameter: Epoch Stage 1 & 2, Batch Size, Learning Rate
- 2-Stage Training: Head Training → Fine-tuning
- Grafik Accuracy & Loss langsung tampil setelah training

### 4. 📝 Riwayat Training
- Lihat grafik accuracy/loss semua model yang sudah dilatih
- Tabel perbandingan best accuracy antar model

### 5. 📊 Evaluasi
- Evaluasi semua model terlatih pada data test
- Metric: Accuracy, Precision, Recall, F1-Score, Inference Time
- Confusion Matrix interaktif
- Classification Report lengkap per kelas
- Otomatis menentukan Model Terbaik

### 6. 🔍 Prediksi
- Upload foto rempah (dari HP, internet, atau dataset baru)
- Prediksi otomatis dengan confidence score
- Tampilkan Top-5 prediksi

---

## 📁 Struktur File
```
app.py              ← File utama Streamlit
requirements.txt    ← Dependensi Python
hasil_rempah/       ← Dibuat otomatis saat running
  ├── dataset/      ← Dataset yang diekstrak
  ├── model/        ← Model .h5 hasil training
  ├── laporan/      ← CSV split, classification report
  ├── grafik/       ← Grafik training
  └── state.json    ← State session (persisten)
```

## ⚙️ Spesifikasi Teknis
- Input: 224×224 piksel, RGB
- Arsitektur: Transfer Learning dari ImageNet
- Optimizer: Adam dengan ReduceLROnPlateau
- Regularisasi: L2 + Dropout + BatchNormalization
- Loss: Categorical Crossentropy + Label Smoothing
- Callbacks: EarlyStopping (patience=4)