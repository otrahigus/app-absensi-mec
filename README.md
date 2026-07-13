# 📸 Sistem Absen Wajah Siswa MEC (Versi Perbaikan)

Sistem absensi otomatis berbasis pengenalan wajah yang bisa diakses dari HP.

## 🔧 Apa yang Diperbaiki

**Penting: data wajah sempat hilang setelah beberapa hari.**
Penyebabnya, embedding wajah sebelumnya disimpan di file lokal (`data/embeddings.pkl`).
Di banyak layanan hosting (Streamlit Cloud, dsb) penyimpanan lokal itu **sementara** —
begitu aplikasi restart/redeploy/tidur-lalu-bangun, file itu ikut terhapus, sehingga
semua wajah yang sudah didaftarkan "hilang" dan sistem minta didaftarkan ulang.

**Perbaikannya:** embedding wajah sekarang disimpan **permanen di Google Sheets**
(worksheet baru bernama `DataWajah`), bukan file lokal. Karena Google Sheets adalah
penyimpanan cloud, datanya **tidak akan hilang** walau server di-restart kapan pun —
jadi benar-benar cukup foto sekali saat daftar, seterusnya tinggal absen foto.

Alur lengkapnya:

| Tahap | Kapan dilakukan | Yang terjadi |
|---|---|---|
| **Daftar Wajah Baru** | **Sekali saja** per siswa | Foto diambil → embedding wajah dihitung → disimpan permanen sebagai baris baru di worksheet `DataWajah` (Google Sheets) |
| **Absen Wajah** | **Setiap hari**, berkali-kali | Foto diambil → data dari `DataWajah` di Google Sheets → dicocokkan → langsung tercatat ke worksheet `Absensi`, **tanpa perlu isi data lagi** |

Jadi siswa yang sudah didaftarkan wajahnya **tidak perlu mendaftar ulang** untuk absen — cukup ambil foto di menu Absen Wajah dan sistem otomatis mengenali siapa dia, kapan pun, walau server sudah restart berkali-kali.

> Catatan: foto referensi (`data/faces/*.jpg`) tetap disimpan lokal hanya untuk tampilan di UI saat itu juga — bukan sumber data pengenalan. Data pengenalan sesungguhnya (embedding) sudah aman di Google Sheets. Kalau butuh foto referensi juga ikut permanen, foto bisa diunggah ke folder Google Drive lewat service account yang sama — beri tahu saya kalau ini diperlukan.

Tambahan lain:
- ✅ Anti-duplikasi: satu siswa hanya tercatat sekali per hari di Google Sheets
- ✅ Menu "Kelola Data Wajah" untuk melihat/menghapus siswa yang sudah terdaftar
- ✅ Filter & unduh rekap absensi (CSV) langsung dari aplikasi

## ✨ Fitur
- 📷 Absen dengan foto wajah langsung dari HP (otomatis, tanpa isi form)
- 📤 Upload atau ambil foto wajah saat pendaftaran
- 📝 Daftar wajah baru (sekali per siswa)
- 📊 Lihat & unduh rekap absensi
- ⚙️ Kelola (hapus/lihat) data wajah terdaftar
- ☁️ Data tersimpan otomatis ke Google Sheets

## 🚀 Cara Pakai

### 1. Instalasi
```bash
pip install -r requirements.txt
```

### 2. Setup Google Sheets
1. Buka [Google Cloud Console](https://console.cloud.google.com/), buat project baru.
2. Aktifkan **Google Sheets API** dan **Google Drive API**.
3. Buat **Service Account**, lalu unduh file kredensial JSON.
4. Buat Google Spreadsheet baru, salin **ID spreadsheet** dari URL-nya.
5. Bagikan (Share) spreadsheet tersebut ke email service account (terlihat di file JSON, field `client_email`) dengan akses **Editor**.
6. Salin `.streamlit/secrets.toml.example` menjadi `.streamlit/secrets.toml`, lalu isi `SPREADSHEET_ID` dan seluruh isi `[gcp_service_account]` sesuai file JSON Anda.

### 3. Jalankan aplikasi
```bash
streamlit run app.py
```

### 4. Alur pemakaian
1. Buka link aplikasi
2. **Admin/guru** mendaftarkan wajah tiap siswa di menu **"Daftar Wajah Baru"** — dilakukan sekali per siswa
3. **Siswa** absen tiap hari di menu **"Absen Wajah"** — tinggal ambil foto, tanpa isi form apapun
4. Lihat rekap di menu **"Lihat Rekap"**

## 📱 Akses
Bisa diakses dari HP maupun PC melalui browser (kamera HP didukung lewat `st.camera_input`).

## 🔧 Teknologi
- Streamlit
- DeepFace (face recognition, model `Facenet`)
- Google Sheets API (`gspread`)
- OpenCV

## 📂 Struktur Proyek
```
absen-wajah-mec/
├── app.py                  # Aplikasi utama Streamlit
├── requirements.txt
├── utils/
│   ├── face_recognition.py # Pendaftaran & pencocokan wajah (embedding lokal)
│   └── sheets.py            # Integrasi Google Sheets
├── data/
│   └── faces/               # Foto referensi siswa terdaftar (hanya untuk tampilan UI,
│                             #   bukan sumber data pengenalan - lihat catatan di atas)
└── .streamlit/
    └── secrets.toml.example
```

## ⚠️ Catatan
- Untuk hasil terbaik, ambil foto pendaftaran dengan pencahayaan cukup dan wajah menghadap kamera.
- Ambang batas kemiripan wajah (`THRESHOLD` di `utils/face_recognition.py`) bisa disesuaikan jika terlalu ketat/longgar.
- Data wajah (embedding) sekarang tersimpan di worksheet **`DataWajah`** pada Google Sheets yang sama dengan absensi — jangan hapus/ubah manual isi kolom `Embedding` di sana, karena itu "otak" pengenalan wajahnya.
- Karena data pengenalan sekarang ada di Google Sheets, hasil pendaftaran akan **tetap ada** meski aplikasi di-restart/redeploy/tidur - tidak perlu daftar ulang lagi.