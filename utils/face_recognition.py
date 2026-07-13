"""
utils/face_recognition.py

Modul ini menangani:
1. Membuat "embedding" (sidik jari wajah) dari foto menggunakan DeepFace
2. Mencocokkan foto absen dengan data wajah yang tersimpan

PERBAIKAN PENTING:
Sebelumnya embedding disimpan di file lokal (data/embeddings.pkl). Di banyak
hosting (Streamlit Cloud, dll) file lokal itu HILANG setiap kali aplikasi
restart/redeploy, sehingga data wajah yang sudah didaftarkan "menghilang"
dan minta didaftarkan ulang.

Sekarang embedding disimpan PERMANEN di Google Sheets (lewat utils/sheets.py),
jadi:
- Pendaftaran wajah tetap cukup SEKALI per siswa
- Data TIDAK hilang walau server restart/redeploy/sleep
- Foto absen berikutnya tinggal dicocokkan dengan data yang sudah ada di Sheets
"""

import os
import numpy as np
from deepface import DeepFace

from utils import sheets as sh

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FACES_DIR = os.path.join(DATA_DIR, "faces")

MODEL_NAME = "Facenet"          # Model DeepFace: cepat & cukup akurat untuk absensi
DETECTOR_BACKEND = "opencv"     # Deteksi wajah ringan, cocok untuk foto dari HP
THRESHOLD = 10.0                # Ambang batas jarak (euclidean) - makin kecil makin mirip

os.makedirs(FACES_DIR, exist_ok=True)


def get_embedding(image_path: str) -> np.ndarray:
    """Hitung embedding wajah dari sebuah file gambar."""
    result = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,
    )
    return np.array(result[0]["embedding"])


def register_face(nis: str, nama: str, kelas: str, image_path: str) -> str:
    """
    Daftarkan wajah baru. Dipanggil HANYA SEKALI per siswa (menu "Daftar Wajah Baru").
    Embedding disimpan PERMANEN ke Google Sheets (bukan file lokal), sehingga
    tidak hilang walau server restart/redeploy.
    """
    existing = sh.cek_nis_terdaftar(nis)
    if existing:
        return f"NIS {nis} sudah terdaftar atas nama {existing['nama']}. Gunakan menu 'Kelola Data Wajah' jika perlu memperbarui foto."

    embedding = get_embedding(image_path)

    # Simpan salinan foto referensi secara lokal (opsional, untuk ditampilkan di UI saja -
    # bukan sumber utama data pengenalan, karena file lokal bisa hilang saat redeploy)
    saved_image_path = os.path.join(FACES_DIR, f"{nis}_{nama}.jpg")
    try:
        with open(image_path, "rb") as src, open(saved_image_path, "wb") as dst:
            dst.write(src.read())
    except OSError:
        pass  # tidak fatal - data pengenalan tetap aman karena sudah ada di Sheets

    sh.simpan_wajah(nis, nama, kelas, embedding)
    return f"✅ Wajah {nama} ({nis}) berhasil didaftarkan dan tersimpan permanen di Google Sheets."


def update_face(nis: str, image_path: str) -> str:
    """Perbarui foto/embedding siswa yang sudah terdaftar (opsional, jika wajah berubah)."""
    existing = sh.cek_nis_terdaftar(nis)
    if not existing:
        return f"NIS {nis} belum terdaftar."

    embedding = get_embedding(image_path)
    sh.update_wajah(nis, embedding)
    return f"✅ Data wajah {existing['nama']} berhasil diperbarui."


def list_registered() -> list:
    """Kembalikan daftar siswa yang sudah terdaftar (diambil langsung dari Google Sheets)."""
    data = sh.ambil_semua_wajah()
    return [
        {"nis": d["nis"], "nama": d["nama"], "kelas": d["kelas"], "terdaftar_pada": d["terdaftar_pada"]}
        for d in data
    ]


def delete_face(nis: str) -> str:
    ok = sh.hapus_wajah(nis)
    if not ok:
        return f"NIS {nis} tidak ditemukan."

    saved_image_path = None
    for f in os.listdir(FACES_DIR):
        if f.startswith(f"{nis}_"):
            saved_image_path = os.path.join(FACES_DIR, f)
            break
    if saved_image_path and os.path.exists(saved_image_path):
        os.remove(saved_image_path)

    return f"🗑️ Data NIS {nis} dihapus."


def recognize_face(image_path: str):
    """
    Cocokkan foto absen dengan semua embedding yang tersimpan di Google Sheets.
    Tidak melakukan pendaftaran apapun - murni pencocokan cepat.

    Return: (nis, nama, kelas, jarak) jika cocok, atau (None, None, None, None) jika tidak ada yang cocok.
    """
    data = sh.ambil_semua_wajah()
    if not data:
        return None, None, None, None

    probe_embedding = get_embedding(image_path)

    best_match, best_dist = None, float("inf")
    for d in data:
        dist = np.linalg.norm(probe_embedding - np.array(d["embedding"]))
        if dist < best_dist:
            best_dist = dist
            best_match = d

    if best_match is not None and best_dist <= THRESHOLD:
        return best_match["nis"], best_match["nama"], best_match["kelas"], best_dist

    return None, None, None, None