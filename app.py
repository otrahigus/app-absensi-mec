"""
Sistem Absensi Wajah - Embedding Manual (tanpa DeepFace.find)
================================================================
- Registrasi: ambil 3-5 foto -> ekstrak embedding -> rata-ratakan -> simpan ke JSON
- Absen: ambil 1 foto -> ekstrak embedding -> bandingkan cosine similarity
         dengan semua embedding tersimpan -> ambil yang paling mirip

Model default: Facenet (ringan, cocok untuk Streamlit Cloud yang RAM-nya terbatas).
Bisa ganti ke ArcFace kalau butuh akurasi lebih tinggi (lebih berat).
"""

import os
import json
import time
import pickle
from datetime import datetime

import numpy as np
import cv2
import streamlit as st
from deepface import DeepFace

# ==============================================================
# KONFIGURASI
# ==============================================================
MODEL_NAME = "Facenet"          # alternatif: "ArcFace" (lebih akurat, lebih berat)
DETECTOR_BACKEND = "opencv"     # ringan & cukup untuk foto HP; alternatif: "retinaface" (lebih akurat, lebih lambat)
EMBEDDING_DB_PATH = "face_embeddings.json"   # database embedding (JSON, mudah dibaca/didebug)
PHOTO_BACKUP_DIR = "face_photos_backup"      # simpan foto asli sebagai cadangan
SIMILARITY_THRESHOLD = 0.55     # cosine similarity minimal supaya dianggap "cocok" (0-1, makin besar makin mirip)
NUM_REGISTRATION_PHOTOS = 3     # jumlah foto saat registrasi (bisa 3-5)

os.makedirs(PHOTO_BACKUP_DIR, exist_ok=True)


# ==============================================================
# UTIL: DATABASE EMBEDDING (JSON)
# ==============================================================
def load_database() -> dict:
    """
    Struktur file JSON:
    {
        "Nama Anak 1": {
            "embedding": [0.123, -0.045, ...],   # rata-rata dari beberapa foto
            "model": "Facenet",
            "registered_at": "2026-07-08 10:00:00",
            "num_photos": 3
        },
        "Nama Anak 2": { ... }
    }
    """
    if not os.path.exists(EMBEDDING_DB_PATH):
        return {}
    with open(EMBEDDING_DB_PATH, "r") as f:
        return json.load(f)


def save_database(db: dict):
    with open(EMBEDDING_DB_PATH, "w") as f:
        json.dump(db, f, indent=2)


# ==============================================================
# UTIL: EKSTRAKSI EMBEDDING
# ==============================================================
def get_embedding_from_image(image_bgr: np.ndarray):
    """
    Ekstrak embedding wajah dari satu gambar (numpy array, format BGR dari OpenCV).
    Mengembalikan (embedding: np.ndarray | None, error_message: str | None)
    """
    try:
        result = DeepFace.represent(
            img_path=image_bgr,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,   # kalau wajah tidak terdeteksi, akan raise error -> kita tangkap
            align=True,
        )
        # DeepFace.represent bisa mengembalikan list (kalau ada >1 wajah terdeteksi)
        if isinstance(result, list):
            if len(result) == 0:
                return None, "Wajah tidak terdeteksi di foto."
            if len(result) > 1:
                return None, "Terdeteksi lebih dari satu wajah. Pastikan hanya 1 wajah di foto."
            embedding = result[0]["embedding"]
        else:
            embedding = result["embedding"]

        return np.array(embedding, dtype=np.float32), None

    except Exception as e:
        return None, f"Gagal mendeteksi wajah: {e}"


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = vec_a / (np.linalg.norm(vec_a) + 1e-10)
    b = vec_b / (np.linalg.norm(vec_b) + 1e-10)
    return float(np.dot(a, b))


# ==============================================================
# FUNGSI UTAMA: REGISTRASI
# ==============================================================
def register_face(name: str, images_bgr: list) -> tuple[bool, str]:
    """
    Daftarkan wajah baru dari beberapa foto (idealnya 3-5 sudut berbeda).
    - Ekstrak embedding dari tiap foto yang valid (wajah terdeteksi)
    - Rata-ratakan embedding-embedding tersebut
    - Simpan ke database JSON + simpan foto asli sebagai cadangan

    Return: (berhasil: bool, pesan: str)
    """
    name = name.strip()
    if not name:
        return False, "Nama tidak boleh kosong."

    embeddings = []
    for idx, img in enumerate(images_bgr):
        emb, err = get_embedding_from_image(img)
        if emb is None:
            # lewati foto yang gagal, tapi catat peringatan
            continue
        embeddings.append(emb)

    if len(embeddings) == 0:
        return False, "Tidak ada foto valid (wajah tidak terdeteksi di semua foto). Coba ulangi dengan pencahayaan lebih baik."

    if len(embeddings) < len(images_bgr):
        skipped = len(images_bgr) - len(embeddings)
        # tetap lanjut, tapi nanti dikasih tahu ke user berapa foto yang terpakai

    # rata-ratakan embedding dari semua foto valid
    avg_embedding = np.mean(embeddings, axis=0)

    db = load_database()
    db[name] = {
        "embedding": avg_embedding.tolist(),
        "model": MODEL_NAME,
        "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_photos": len(embeddings),
    }
    save_database(db)

    # simpan foto asli sebagai cadangan (opsional tapi berguna untuk audit)
    person_dir = os.path.join(PHOTO_BACKUP_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    for idx, img in enumerate(images_bgr):
        cv2.imwrite(os.path.join(person_dir, f"{int(time.time())}_{idx}.jpg"), img)

    return True, f"Berhasil daftarkan '{name}' menggunakan {len(embeddings)} dari {len(images_bgr)} foto."


# ==============================================================
# FUNGSI UTAMA: ABSEN / PENGENALAN
# ==============================================================
def recognize_face(image_bgr: np.ndarray, threshold: float = SIMILARITY_THRESHOLD):
    """
    Kenali wajah dari satu foto dengan membandingkan ke semua embedding tersimpan.
    Pengganti DeepFace.find() -> tidak perlu scan folder, cukup hitung cosine similarity
    terhadap embedding yang sudah ada di memori/JSON.

    Return: dict {
        "recognized": bool,
        "name": str | None,
        "similarity": float,
        "message": str
    }
    """
    db = load_database()
    if len(db) == 0:
        return {"recognized": False, "name": None, "similarity": 0.0,
                "message": "Database wajah masih kosong. Daftarkan wajah dulu."}

    query_embedding, err = get_embedding_from_image(image_bgr)
    if query_embedding is None:
        return {"recognized": False, "name": None, "similarity": 0.0, "message": err}

    best_name = None
    best_similarity = -1.0

    for name, data in db.items():
        stored_embedding = np.array(data["embedding"], dtype=np.float32)
        sim = cosine_similarity(query_embedding, stored_embedding)
        if sim > best_similarity:
            best_similarity = sim
            best_name = name

    if best_similarity >= threshold:
        return {
            "recognized": True,
            "name": best_name,
            "similarity": best_similarity,
            "message": f"Wajah dikenali sebagai '{best_name}' (similarity: {best_similarity:.3f})",
        }
    else:
        return {
            "recognized": False,
            "name": best_name,   # kandidat terdekat, untuk info/debug saja
            "similarity": best_similarity,
            "message": f"Wajah tidak dikenali (similarity tertinggi hanya {best_similarity:.3f}, "
                       f"di bawah threshold {threshold:.2f}).",
        }


# ==============================================================
# UTIL: GOOGLE SHEETS (catat absen)
# ==============================================================
def get_gsheet_client():
    """
    Menggunakan service account yang disimpan di st.secrets.
    Contoh isi .streamlit/secrets.toml:

    [gcp_service_account]
    type = "service_account"
    project_id = "..."
    private_key_id = "..."
    private_key = "..."
    client_email = "..."
    client_id = "..."
    ...

    sheet_url = "https://docs.google.com/spreadsheets/d/xxxxx/edit"
    """
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scopes
    )
    client = gspread.authorize(creds)
    return client


def catat_absen_ke_sheet(name: str, similarity: float):
    """Tambahkan baris absen baru ke Google Sheets. Cegah duplikat di hari yang sama (opsional)."""
    try:
        client = get_gsheet_client()
        sheet = client.open_by_url(st.secrets["sheet_url"]).sheet1

        today_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H:%M:%S")

        # opsional: cek supaya tidak dobel absen di hari yang sama
        records = sheet.get_all_records()
        already_present = any(
            r.get("Nama") == name and r.get("Tanggal") == today_str for r in records
        )
        if already_present:
            return False, f"'{name}' sudah absen hari ini ({today_str})."

        sheet.append_row([name, today_str, time_str, f"{similarity:.3f}"])
        return True, f"Absen '{name}' tercatat pukul {time_str}."

    except Exception as e:
        return False, f"Gagal mencatat ke Google Sheets: {e}"


# ==============================================================
# HELPER: konversi foto Streamlit camera_input -> numpy BGR
# ==============================================================
def camera_input_to_bgr(camera_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img_bgr


# ==============================================================
# STREAMLIT UI
# ==============================================================
def main():
    st.set_page_config(page_title="Absensi Wajah", page_icon="📸")
    st.title("📸 Sistem Absensi Wajah")

    menu = st.sidebar.radio("Menu", ["Daftar Wajah Baru", "Absen Wajah", "Lihat Database"])

    # ------------------------------------------------------------
    if menu == "Daftar Wajah Baru":
        st.header("Daftar Wajah Baru")
        st.caption(
            f"Ambil {NUM_REGISTRATION_PHOTOS} foto dari sudut sedikit berbeda "
            "(hadap depan, miring kiri, miring kanan) untuk akurasi lebih baik."
        )

        name = st.text_input("Nama lengkap anak")

        captured_images = []
        for i in range(NUM_REGISTRATION_PHOTOS):
            cam_photo = st.camera_input(f"Foto ke-{i + 1}", key=f"reg_cam_{i}")
            if cam_photo is not None:
                captured_images.append(camera_input_to_bgr(cam_photo))

        if st.button("Simpan Registrasi", type="primary"):
            if not name.strip():
                st.error("Isi nama dulu.")
            elif len(captured_images) == 0:
                st.error("Ambil minimal 1 foto (idealnya semua foto diisi).")
            else:
                with st.spinner("Memproses wajah..."):
                    ok, msg = register_face(name, captured_images)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    # ------------------------------------------------------------
    elif menu == "Absen Wajah":
        st.header("Absen Wajah")

        threshold = st.slider(
            "Threshold kemiripan (semakin tinggi = semakin ketat)",
            min_value=0.30, max_value=0.90, value=SIMILARITY_THRESHOLD, step=0.01,
        )

        cam_photo = st.camera_input("Ambil foto untuk absen")

        if cam_photo is not None:
            img_bgr = camera_input_to_bgr(cam_photo)
            with st.spinner("Mengenali wajah..."):
                result = recognize_face(img_bgr, threshold=threshold)

            if result["recognized"]:
                st.success(result["message"])
                ok, sheet_msg = catat_absen_ke_sheet(result["name"], result["similarity"])
                if ok:
                    st.info(sheet_msg)
                else:
                    st.warning(sheet_msg)
            else:
                st.error(result["message"])
                if result["name"]:
                    st.caption(f"(Kandidat terdekat: {result['name']}, tapi di bawah threshold)")

    # ------------------------------------------------------------
    elif menu == "Lihat Database":
        st.header("Database Wajah Terdaftar")
        db = load_database()
        if len(db) == 0:
            st.info("Belum ada wajah terdaftar.")
        else:
            rows = [
                {"Nama": name, "Model": d["model"], "Jumlah Foto": d["num_photos"],
                 "Terdaftar": d["registered_at"]}
                for name, d in db.items()
            ]
            st.table(rows)


if __name__ == "__main__":
    main()