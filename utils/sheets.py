"""
utils/sheets.py

Modul integrasi Google Sheets.

PENTING (perbaikan terbaru):
Sebelumnya data wajah (embedding hasil "latih model") disimpan di file lokal
(data/embeddings.pkl). Di banyak layanan hosting (mis. Streamlit Cloud),
penyimpanan lokal itu BERSIFAT SEMENTARA - begitu aplikasi restart/redeploy/
sleep-lalu-bangun, file lokal ikut hilang, sehingga data wajah yang sudah
didaftarkan "hilang" dan minta didaftarkan ulang.

Perbaikannya: embedding wajah sekarang disimpan permanen di Google Sheets
(worksheet terpisah bernama "DataWajah"), bukan di file lokal. Karena
Google Sheets adalah penyimpanan cloud, datanya TIDAK akan hilang walau
server di-restart, redeploy, atau tidur/bangun kapan saja.

Menyimpan setiap absensi sebagai baris baru, dan mencegah SISWA YANG SAMA
tercatat dua kali di HARI YANG SAMA (menghindari duplikasi saat kamera
mendeteksi wajah berkali-kali dalam sesi yang sama).

Setup yang dibutuhkan (lihat README):
1. Buat Service Account di Google Cloud Console, aktifkan Google Sheets API & Drive API.
2. Download file JSON credential, simpan sebagai `service_account.json`
   atau masukkan isinya ke st.secrets["gcp_service_account"].
3. Share Google Sheet tujuan ke email service account (client_email) dengan akses Editor.
4. Isi SPREADSHEET_ID di bawah atau lewat secrets.
"""

import streamlit as st
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

SHEET_NAME = "Absensi"  # nama tab/worksheet untuk log absensi
HEADER = ["Tanggal", "Jam", "NIS", "Nama", "Kelas", "Status"]

FACE_SHEET_NAME = "DataWajah"  # nama tab/worksheet untuk data wajah terdaftar (permanen)
FACE_HEADER = ["NIS", "Nama", "Kelas", "Embedding", "TanggalDaftar"]


class SecretsBelumDiatur(Exception):
    """Dilempar saat Secrets (SPREADSHEET_ID / gcp_service_account) belum diisi di Streamlit Cloud."""
    pass


def _pastikan_secrets_ada():
    """
    Cek Secrets dengan aman TANPA membuat Streamlit melempar exception mentah
    (StreamlitAPIException: No secrets found) saat file/menu Secrets kosong.
    Kalau belum lengkap, lempar error kita sendiri yang pesannya jelas dan
    bisa ditangkap+ditampilkan rapi di app.py.
    """
    try:
        ada_spreadsheet_id = bool(st.secrets.get("SPREADSHEET_ID"))
        ada_service_account = "gcp_service_account" in st.secrets
    except Exception:
        # Terjadi kalau belum ada Secrets sama sekali (file/menu kosong total)
        ada_spreadsheet_id = False
        ada_service_account = False

    if not ada_spreadsheet_id or not ada_service_account:
        raise SecretsBelumDiatur(
            "Secrets belum lengkap. Buka menu app di Streamlit Cloud → titik tiga (⋮) → "
            "'Settings' → 'Secrets', lalu isi 'SPREADSHEET_ID' dan blok '[gcp_service_account]' "
            "sesuai panduan di README.md bagian 'Setup Google Sheets'."
        )


@st.cache_resource(show_spinner=False)
def _get_client():
    """Otentikasi ke Google Sheets. Menggunakan cache agar tidak login berulang kali."""
    _pastikan_secrets_ada()
    if "gcp_service_account" in st.secrets:
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    else:
        creds = Credentials.from_service_account_file("service_account.json", scopes=SCOPES)
    return gspread.authorize(creds)


def _get_worksheet():
    client = _get_client()
    spreadsheet_id = st.secrets.get("SPREADSHEET_ID", None)
    if not spreadsheet_id:
        raise SecretsBelumDiatur("SPREADSHEET_ID belum diatur di Secrets.")

    sh = client.open_by_key(spreadsheet_id)
    try:
        ws = sh.worksheet(SHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=SHEET_NAME, rows=1000, cols=len(HEADER))
        ws.append_row(HEADER)

    # Pastikan header selalu ada di baris pertama
    first_row = ws.row_values(1)
    if first_row != HEADER:
        ws.insert_row(HEADER, 1)

    return ws


def already_checked_in_today(nis: str) -> bool:
    """Cek apakah NIS ini sudah absen hari ini, untuk mencegah baris duplikat."""
    ws = _get_worksheet()
    records = ws.get_all_records()
    today = datetime.now().strftime("%Y-%m-%d")
    for r in records:
        if str(r.get("NIS")) == str(nis) and r.get("Tanggal") == today:
            return True
    return False


def catat_absen(nis: str, nama: str, kelas: str, status: str = "Hadir") -> str:
    """Tambahkan satu baris absensi baru ke Google Sheets."""
    if already_checked_in_today(nis):
        return f"⚠️ {nama} sudah tercatat absen hari ini."

    ws = _get_worksheet()
    now = datetime.now()
    ws.append_row([
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        nis,
        nama,
        kelas,
        status,
    ])
    return f"✅ Absensi {nama} tercatat pukul {now.strftime('%H:%M:%S')}."


def ambil_rekap() -> list:
    """Ambil seluruh data rekap absensi sebagai list of dict."""
    ws = _get_worksheet()
    return ws.get_all_records()


# =========================================================================
# PENYIMPANAN DATA WAJAH (embedding) - PERMANEN di Google Sheets
# =========================================================================

def _get_face_worksheet():
    client = _get_client()
    spreadsheet_id = st.secrets.get("SPREADSHEET_ID", None)
    if not spreadsheet_id:
        raise SecretsBelumDiatur("SPREADSHEET_ID belum diatur di Secrets.")

    sh = client.open_by_key(spreadsheet_id)
    try:
        ws = sh.worksheet(FACE_SHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=FACE_SHEET_NAME, rows=1000, cols=len(FACE_HEADER))
        ws.append_row(FACE_HEADER)

    first_row = ws.row_values(1)
    if first_row != FACE_HEADER:
        ws.insert_row(FACE_HEADER, 1)

    return ws


@st.cache_data(ttl=30, show_spinner=False)
def ambil_semua_wajah() -> list:
    """
    Ambil semua data wajah terdaftar dari Google Sheets (bukan file lokal).
    Di-cache 30 detik supaya proses absen tetap cepat (tidak memanggil API
    Google tiap kali kamera mengambil foto), tapi tetap otomatis ter-refresh
    berkala dan setelah ada pendaftaran baru (lihat `bersihkan_cache_wajah`).
    """
    ws = _get_face_worksheet()
    rows = ws.get_all_records()
    hasil = []
    for r in rows:
        embedding_str = r.get("Embedding", "")
        if not embedding_str:
            continue
        embedding = [float(x) for x in embedding_str.split(",")]
        hasil.append({
            "nis": str(r["NIS"]),
            "nama": r["Nama"],
            "kelas": r["Kelas"],
            "embedding": embedding,
            "terdaftar_pada": r.get("TanggalDaftar", ""),
        })
    return hasil


def bersihkan_cache_wajah():
    """Panggil ini setelah daftar/hapus wajah supaya data terbaru langsung terpakai."""
    ambil_semua_wajah.clear()


def cek_nis_terdaftar(nis: str):
    """Cek apakah NIS sudah ada di Google Sheets. Kembalikan data siswa jika ada, None jika belum."""
    for w in ambil_semua_wajah():
        if w["nis"] == str(nis):
            return w
    return None


def simpan_wajah(nis: str, nama: str, kelas: str, embedding) -> None:
    """Simpan embedding wajah baru sebagai baris permanen di Google Sheets."""
    ws = _get_face_worksheet()
    embedding_str = ",".join(str(float(x)) for x in embedding)
    ws.append_row([str(nis), nama, kelas, embedding_str, datetime.now().isoformat()])
    bersihkan_cache_wajah()


def update_wajah(nis: str, embedding) -> bool:
    """Perbarui embedding wajah siswa yang sudah terdaftar."""
    ws = _get_face_worksheet()
    cell = ws.find(str(nis), in_column=1)
    if cell is None:
        return False
    embedding_str = ",".join(str(float(x)) for x in embedding)
    ws.update_cell(cell.row, FACE_HEADER.index("Embedding") + 1, embedding_str)
    ws.update_cell(cell.row, FACE_HEADER.index("TanggalDaftar") + 1, datetime.now().isoformat())
    bersihkan_cache_wajah()
    return True


def hapus_wajah(nis: str) -> bool:
    """Hapus baris data wajah siswa dari Google Sheets berdasarkan NIS."""
    ws = _get_face_worksheet()
    cell = ws.find(str(nis), in_column=1)
    if cell is None:
        return False
    ws.delete_rows(cell.row)
    bersihkan_cache_wajah()
    return True