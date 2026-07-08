# 📸 Sistem Absen Wajah Siswa MEC

Sistem absensi otomatis berbasis pengenalan wajah yang bisa diakses dari HP.

Cara kerja: **daftar wajah sekali saja** (5 foto otomatis, langsung dilatih), setelah itu absen berikutnya tinggal ambil 1 foto — tidak perlu daftar ulang.

## ✨ Fitur
- 📷 Absen dengan foto wajah langsung dari HP
- 📝 Daftar wajah baru (5 foto otomatis, model auto-terlatih setelah selesai)
- 📊 Lihat rekap absensi + unduh CSV
- ☁️ Data tersimpan otomatis ke Google Sheets

## 🚀 Cara Pakai
1. Buka link aplikasi
2. Daftar wajah di menu **"Daftar Wajah Baru"** — isi nama, lalu ambil 5 foto berturut-turut (hadap depan, miring kiri, miring kanan). Setelah foto ke-5, model otomatis dilatih.
3. Absen di menu **"Absen Wajah"** — cukup ambil 1 foto.
4. Lihat rekap di menu **"Lihat Rekap"**.

## 📱 Akses
Bisa diakses dari HP maupun PC melalui browser.

> **Catatan kamera**: `st.camera_input` butuh HTTPS untuk akses kamera browser (kecuali di `localhost`). Kalau diakses dari HP lewat IP lokal (`http://192.168.x.x:8501`), kamera tidak akan muncul. Untuk tes dari HP, deploy dulu ke Streamlit Cloud (otomatis HTTPS).

## 🔧 Teknologi
- **Streamlit** — antarmuka web
- **OpenCV (LBPH Face Recognizer)** — deteksi & pengenalan wajah, ringan dan tanpa model deep-learning besar
- **streamlit-gsheets-connection** — koneksi ke Google Sheets
- Python: `opencv-contrib-python-headless`, `numpy`, `pandas`, `Pillow`

---

## Instalasi Lokal

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501`.

---

## Setup Google Sheets (`streamlit-gsheets-connection`)

Library ini beda formatnya dari `gspread`. Langkahnya:

### 1. Buat Google Sheet
Buat spreadsheet baru, worksheet pertama beri header:

| name | date | time |
|------|------|------|

### 2. Buat Service Account
1. Buka [Google Cloud Console](https://console.cloud.google.com/) → buat/pilih project.
2. Aktifkan **Google Sheets API** dan **Google Drive API**.
3. **IAM & Admin → Service Accounts** → buat service account baru → buat key **JSON** → unduh.
4. Buka Google Sheet-nya → **Share** → tambahkan email service account (`xxx@xxx.iam.gserviceaccount.com`) sebagai **Editor**.

### 3. Isi `.streamlit/secrets.toml`

```toml
[connections.gsheets]
spreadsheet = "https://docs.google.com/spreadsheets/d/xxxxxxxxxxxx/edit"
type = "service_account"
project_id = "nama-project-kamu"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "xxx@xxx.iam.gserviceaccount.com"
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
```

**Penting:**
- Field-field ini semua masuk ke satu section `[connections.gsheets]` (beda dengan `gspread` yang pakai `[gcp_service_account]` terpisah).
- `private_key` harus satu baris dengan `\n` literal persis seperti di file JSON asli.
- **Jangan commit `secrets.toml` ke git.**

Tambahkan `.streamlit/secrets.toml` ke `.gitignore`.

---

## Deploy ke Streamlit Cloud

1. Push ke GitHub — **tanpa** `secrets.toml`, `dataset/`, `trainer/`.
2. [share.streamlit.io](https://share.streamlit.io) → **New app** → pilih repo & `app.py`.
3. **Advanced settings → Secrets** → tempel isi `secrets.toml` di atas.
4. Deploy.

### `.gitignore` disarankan
```
venv/
__pycache__/
.streamlit/secrets.toml
dataset/
trainer/
```

> **Filesystem Streamlit Cloud tidak persisten** — `dataset/` dan `trainer/trainer.yml` bisa hilang saat app redeploy/restart. Untuk pemakaian jangka panjang, kamu perlu strategi backup (misal upload folder dataset ke Google Drive secara berkala, atau pindah ke storage eksternal). Untuk uji coba/skala kecil di kampung, ini biasanya masih bisa diterima asal tidak sering redeploy.

---

## Tentang Akurasi (penting dibaca)

Sistem ini pakai **LBPH (Local Binary Patterns Histograms)** — metode klasik OpenCV, bukan deep learning. Kelebihannya: ringan, cepat diinstal, tidak butuh download model besar, cocok untuk hosting gratisan.

Trade-off-nya: akurasi LBPH **lebih rentan** terhadap perubahan pencahayaan, sudut wajah, dan kualitas kamera HP dibanding metode embedding berbasis deep learning (Facenet/ArcFace). Untuk 10–50 orang dengan foto yang cukup jelas & pencahayaan konsisten, LBPH biasanya masih memadai. Tips supaya akurasi maksimal:

- Ambil foto registrasi di lokasi/pencahayaan yang mirip dengan tempat absen nanti.
- Kalau sering salah kenali, coba **naikkan** ambang confidence di menu Absen (lebih ketat) — konsekuensinya kadang wajah yang benar perlu difoto ulang.
- Kalau sering menolak wajah yang benar, **turunkan** ambang confidence.
- Tambah foto registrasi (menu "Daftar Wajah Baru" bisa dipanggil ulang untuk nama yang sama — foto baru ditambahkan, bukan menimpa) untuk orang yang sering gagal dikenali.

Kalau nanti butuh akurasi lebih tinggi dan instalasi `tensorflow`/`deepface` bisa diatasi, sistem bisa di-upgrade ke pendekatan embedding (Facenet/ArcFace) yang jauh lebih akurat untuk variasi pencahayaan & sudut.

---

## Troubleshooting

| Masalah | Kemungkinan Penyebab |
|---|---|
| `cv2.face` tidak ditemukan (`AttributeError: module 'cv2' has no attribute 'face'`) | Salah install `opencv-python` biasa. Harus `opencv-contrib-python` atau `opencv-contrib-python-headless` (sudah benar di requirements.txt ini) — jangan install `opencv-python` polos di environment yang sama, bisa bentrok. |
| Kamera tidak muncul di HP | Diakses lewat HTTP bukan HTTPS. |
| "Wajah tidak terdeteksi" terus | Pencahayaan kurang, wajah terlalu jauh/kecil (di bawah `MIN_FACE_SIZE`), atau wajah miring >45°. |
| Absen selalu "tidak dikenali" walau sudah daftar | Turunkan... eh, **naikkan** ambang confidence di slider (nilai LBPH: makin kecil = makin ketat, makin besar = makin longgar). |
| Salah kenali orang lain | Turunkan ambang confidence (lebih ketat), atau tambah foto registrasi dengan variasi sudut/cahaya. |
| Gagal mencatat ke Google Sheets | Cek format `secrets.toml` (section `[connections.gsheets]`), dan pastikan email service account sudah Editor di Sheet. |
| Data wajah hilang setelah redeploy | Filesystem Streamlit Cloud tidak persisten — lihat catatan di bagian Deploy. |

---

## Struktur File

```
.
├── app.py
├── requirements.txt
├── dataset/                  # foto wajah per orang (dibuat otomatis)
│   └── <nama>/
│       └── 1.jpg, 2.jpg, ...
├── trainer/
│   ├── trainer.yml           # model LBPH terlatih
│   └── labels.csv            # mapping label -> nama
└── .streamlit/
    └── secrets.toml          # kredensial Google Sheets (buat manual, JANGAN commit)
```

## Privasi
Foto dan data wajah anak-anak bersifat sensitif. Pastikan mendapat izin orang tua/wali sebelum menyimpan foto, dan jangan expose folder `dataset/` ke publik.