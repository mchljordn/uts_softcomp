# Panduan Penggunaan Aplikasi
## Prediksi Risiko Dropout Mahasiswa
### Manual FIS vs GA Tuning vs Neuro-Fuzzy ANN

---

## 1. Ringkasan Aplikasi

Aplikasi ini membandingkan tiga pendekatan:
- Tahap 1: Manual FIS (berbasis intuisi pakar)
- Tahap 2: Evolutionary Tuning dengan Genetic Algorithm (GA)
- Tahap 3: Neural Tuning dengan Neuro-Fuzzy ANN

Antarmuka utama menggunakan Streamlit agar pengguna dapat:
- memasukkan parameter mahasiswa,
- melihat skor dan label risiko,
- memvisualisasikan membership function,
- membandingkan performa evaluasi tiap metode.

---

## 2. Persyaratan Sistem

### Minimum
- Python 3.9 atau lebih baru

### Library (mengacu ke requirements.txt)
- numpy>=1.24
- scikit-fuzzy>=0.4.2
- matplotlib>=3.7
- pandas>=2.0
- scipy>=1.10
- streamlit>=1.30
- pygad>=3.0.0
- torch>=2.0.0

---

## 3. Struktur Folder Inti

```text
uts/
├── app.py                 # UI lama (Tahap 1 saja)
├── app-copy.py            # UI utama terintegrasi (Tahap 1, 2, 3)
├── fis_manual.py          # Manual FIS (MF, rules, evaluasi)
├── fis_ga.py              # Optimasi GA untuk tuning parameter MF
├── fis_ann.py             # Neuro-Fuzzy ANN untuk tuning/prediksi
├── requirements.txt       # Daftar dependency Python
├── data/
│   └── data.csv           # Dataset referensi
├── notebooks/
│   └── TAHAP1.ipynb       # Notebook pendukung
├── docs/
│   └── PANDUAN.md         # Dokumen ini
└── archive/               # Arsip dokumen/berkas lama
```

---

## 4. Instalasi

Lakukan dari root project (folder `uts`).

### Windows (PowerShell)
```bash
cd "c:\coding\sem 6\soft computing\uts"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux/Mac
```bash
cd /path/to/uts
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 5. Menjalankan Aplikasi

### Penting (Windows)
- Hindari menjalankan aplikasi dengan `py -m streamlit ...` jika launcher `py` menunjuk ke Python global.
- Gunakan interpreter dari folder `.venv` agar semua dependency terbaca konsisten.

### A. Aplikasi utama (disarankan)
```bash
.venv\Scripts\python.exe -m streamlit run app-copy.py
```

Buka browser di:
- http://localhost:8501

### B. Aplikasi sederhana (manual FIS saja)
```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

### C. Cara cepat (Windows)
```bash
powershell -ExecutionPolicy Bypass -File .\start_app.ps1
```

---

## 6. Fungsi Utama Tiap Tab (app-copy.py)

### Tab 1 - Prediksi
- Input 4 parameter:
   - IPK Semester (0-4)
   - Tingkat Kehadiran (0-100)
   - Jumlah MK Gagal (0-10)
   - Status Ekonomi (0-1)
- Klik tombol hitung.
- Sistem menampilkan hasil tiga metode (Manual, GA, ANN) secara berdampingan.

### Tab 2 - Membership Functions
- Menampilkan MF Manual FIS untuk semua variabel input dan output.
- Berguna untuk melihat desain kurva berbasis intuisi pakar.

### Tab 3 - Evaluasi dan Perbandingan
- Menjalankan evaluasi batch pada data simulasi UCI.
- Menampilkan akurasi, confusion matrix, dan ringkasan komparatif metode.

### Tab 4 - Tahap 2: Neuro-Fuzzy
- Melatih model ANN dari data hasil FIS.
- Parameter pelatihan dapat diatur (epochs dan learning rate).
- Menampilkan kurva training dan metrik hasil.

### Tab 5 - Tahap 3: GA Tuning
- Menjalankan optimasi GA untuk tuning parameter MF.
- Parameter eksperimen dapat diubah (population size dan generations).
- Menampilkan evolusi fitness dan distribusi populasi tiap generasi.

### Tab 6 - Perbandingan MF
- Membandingkan kurva MF antara Manual, GA, dan ANN.
- Mendukung analisis pergeseran kurva hasil optimasi.

---

## 7. Interpretasi Skor Risiko

| Skor Risiko | Label  | Rekomendasi |
|---|---|---|
| 0 - 39 | Rendah | Monitoring rutin |
| 40 - 64 | Sedang | Konseling akademik |
| 65 - 100 | Tinggi | Intervensi segera |

---

## 8. Uji Cepat (Checklist)

Setelah aplikasi terbuka:
1. Coba input nilai di Tab Prediksi lalu pastikan skor Manual muncul.
2. Jalankan Tab Neuro-Fuzzy, tunggu training selesai, lalu cek skor ANN muncul di Tab Prediksi.
3. Jalankan Tab GA Tuning, tunggu optimasi selesai, lalu cek skor GA muncul di Tab Prediksi.
4. Jalankan Tab Evaluasi dan pastikan grafik/matriks tampil.
5. Buka Tab Perbandingan MF untuk verifikasi visual pergeseran kurva.

---

## 9. Troubleshooting

### Streamlit tidak ditemukan
```bash
pip install streamlit
```

### Error module tidak ditemukan (misal torch, pygad, skfuzzy)
```bash
pip install -r requirements.txt
```

### Port 8501 sedang dipakai
```bash
.venv\Scripts\python.exe -m streamlit run app-copy.py --server.port 8502
```

### Masih ter-import dari Python global (contoh path `C:\Users\...\pythoncore-...`)
- Pastikan command diawali `.venv\Scripts\python.exe`.
- Cek interpreter aktif:

```bash
.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
```

Output harus mengarah ke folder project ini, misalnya:
- `C:\coding\sem 6\soft computing\uts\.venv\Scripts\python.exe`

### Prediksi GA/ANN belum muncul di Tab Prediksi
- Pastikan proses training ANN di Tab 4 sudah dijalankan.
- Pastikan proses GA tuning di Tab 5 sudah selesai dijalankan.

---

## 10. Catatan

- `app-copy.py` adalah aplikasi utama untuk pengumpulan UTS.
- `app.py` dipertahankan sebagai versi sederhana Tahap 1.
- Hasil GA/ANN dapat sedikit berbeda antar-run karena proses optimasi berbasis inisialisasi dan iterasi.
