# Panduan Penggunaan Aplikasi
## Prediksi Risiko Dropout Mahasiswa — Tahap 1 Manual FIS

---

## Persyaratan Sistem

| Komponen | Versi Minimum |
|---|---|
| Python | 3.9+ |
| scikit-fuzzy | 0.4.2 |
| numpy | 1.24 |
| matplotlib | 3.7 |
| streamlit | 1.30 |
| scipy | 1.10 |

---

## Instalasi

```bash
# 1. Clone / salin folder proyek
cd fis_dropout/

# 2. (Opsional) Buat virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Menjalankan Aplikasi

### Cara 1 — Streamlit (Antarmuka Web)
```bash
streamlit run app.py
```
Buka browser di: **http://localhost:8501**

### Cara 2 — Script Python langsung
```bash
python fis_manual.py
```
Output: teks di terminal + file gambar `mf_tahap1.png`, `eval_tahap1.png`

---

## Fitur Aplikasi (Streamlit)

### Tab 1 — Prediksi
1. Atur nilai slider:
   - **IPK Semester** (0–4.0)
   - **Tingkat Kehadiran** (0–100%)
   - **Jumlah MK Gagal** (0–10)
   - **Status Ekonomi** (0=Rentan, 1=Stabil)
2. Klik tombol **Hitung Risiko Dropout**
3. Lihat skor, kategori, dan rekomendasi intervensi

### Tab 2 — Membership Functions
- Visualisasi kurva MF semua variabel input dan output

### Tab 3 — Evaluasi Batch
- Klik **Jalankan Evaluasi** untuk menguji 120 sampel simulasi
- Lihat confusion matrix dan akurasi per kelas

---

## Struktur File

```
fis_dropout/
├── fis_manual.py      # Core FIS: MF, Rules, Inferensi, Evaluasi
├── app.py             # Aplikasi Streamlit (UI)
├── requirements.txt   # Daftar library
└── PANDUAN.md         # File ini
```

---

## Interpretasi Output

| Skor Risiko | Label  | Tindakan                        |
|---|---|---|
| 0 – 39      | Rendah | Monitoring rutin                 |
| 40 – 64     | Sedang | Konseling akademik               |
| 65 – 100    | Tinggi | Intervensi segera (dosen wali)   |
