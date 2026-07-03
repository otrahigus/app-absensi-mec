import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from PIL import Image
from streamlit_gsheets import GSheetsConnection

# ------------------------------------------------------------------
# PATH & SETUP
# ------------------------------------------------------------------
DATASET_DIR = "dataset"
TRAINER_PATH = "trainer/trainer.yml"
LABELS_PATH = "trainer/labels.csv"
GSHEET_WORKSHEET = "Sheet1"  # sesuaikan kalau nama sheet-mu beda

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs("trainer", exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------
def get_face(image_bgr):
    """Deteksi wajah pertama pada gambar, kembalikan crop grayscale-nya."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    face = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
    return face


def train_model():
    """Latih ulang model LBPH dari semua foto di folder dataset/."""
    faces, labels = [], []
    label_map = {}
    current_label = 0

    for person_name in sorted(os.listdir(DATASET_DIR)):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        label_map[current_label] = person_name
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(current_label)
        current_label += 1

    if len(faces) == 0:
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(TRAINER_PATH)

    pd.DataFrame(list(label_map.items()), columns=["label", "name"]).to_csv(
        LABELS_PATH, index=False
    )
    return True


def load_model():
    """Muat model yang sudah dilatih, kalau ada."""
    if not os.path.exists(TRAINER_PATH) or not os.path.exists(LABELS_PATH):
        return None, None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)
    labels_df = pd.read_csv(LABELS_PATH)
    label_map = dict(zip(labels_df["label"], labels_df["name"]))
    return recognizer, label_map


def get_gsheets_connection():
    return st.connection("gsheets", type=GSheetsConnection)


def read_attendance():
    """Baca semua data absensi dari Google Sheets."""
    conn = get_gsheets_connection()
    df = conn.read(worksheet=GSHEET_WORKSHEET, ttl=0)
    df = df.dropna(how="all")
    if df.empty:
        df = pd.DataFrame(columns=["name", "date", "time"])
    return df


def mark_attendance(name):
    """Catat kehadiran ke Google Sheets; return False kalau sudah absen hari ini."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    conn = get_gsheets_connection()
    df = read_attendance()

    already = ((df["name"] == name) & (df["date"] == date_str)).any()
    if already:
        return False

    new_row = pd.DataFrame([[name, date_str, time_str]], columns=["name", "date", "time"])
    updated_df = pd.concat([df, new_row], ignore_index=True)
    conn.update(worksheet=GSHEET_WORKSHEET, data=updated_df)
    return True


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.set_page_config(page_title="Absensi Wajah", page_icon="🧑‍💻")
st.title("🧑‍💻 Sistem Absensi Wajah")

menu = st.sidebar.radio("Menu", ["Daftar Wajah Baru", "Absen Sekarang", "Rekap Absensi"])

# ---------------- DAFTAR WAJAH BARU ----------------
if menu == "Daftar Wajah Baru":
    st.header("Daftar Wajah Baru")
    name = st.text_input("Nama lengkap")
    st.caption("Ambil 5 foto dengan sudut/pencahayaan sedikit berbeda untuk hasil terbaik.")

    photo = st.camera_input("Ambil foto wajah")

    if photo and name:
        person_dir = os.path.join(DATASET_DIR, name.strip().replace(" ", "_"))
        os.makedirs(person_dir, exist_ok=True)

        img = Image.open(photo)
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        face = get_face(img_bgr)

        if face is None:
            st.warning("Wajah tidak terdeteksi. Coba lagi dengan pencahayaan lebih baik.")
        else:
            count = len(os.listdir(person_dir)) + 1
            save_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(save_path, face)
            st.success(f"Foto ke-{count} tersimpan untuk **{name}**.")

    st.divider()
    if st.button("🔄 Latih Model Sekarang", type="primary"):
        with st.spinner("Melatih model..."):
            ok = train_model()
        if ok:
            st.success("Model berhasil dilatih! Sekarang bisa absen di menu 'Absen Sekarang'.")
        else:
            st.error("Belum ada data wajah tersimpan. Silakan ambil foto dulu di atas.")

# ---------------- ABSEN SEKARANG ----------------
elif menu == "Absen Sekarang":
    st.header("Absen Sekarang")
    recognizer, label_map = load_model()

    if recognizer is None:
        st.warning(
            "Model belum dilatih. Buka menu 'Daftar Wajah Baru', daftarkan wajah, "
            "lalu klik 'Latih Model Sekarang'."
        )
    else:
        photo = st.camera_input("Ambil foto untuk absen")
        if photo:
            img = Image.open(photo)
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            face = get_face(img_bgr)

            if face is None:
                st.warning("Wajah tidak terdeteksi. Coba lagi.")
            else:
                label, confidence = recognizer.predict(face)
                # LBPH: confidence makin RENDAH = makin mirip
                if confidence < 70:
                    name = label_map.get(label, "Unknown")
                    marked = mark_attendance(name)
                    if marked:
                        st.success(f"✅ Absen berhasil: **{name}** (confidence: {confidence:.1f})")
                    else:
                        st.info(f"{name} sudah absen hari ini.")
                else:
                    st.error(
                        f"Wajah tidak dikenali (confidence: {confidence:.1f}). "
                        "Coba lagi atau daftarkan wajah dulu."
                    )

# ---------------- REKAP ABSENSI ----------------
elif menu == "Rekap Absensi":
    st.header("Rekap Absensi")
    df = read_attendance()
    if not df.empty:
        st.dataframe(df.sort_values(by=["date", "time"], ascending=False), use_container_width=True)
        st.download_button(
            "⬇️ Unduh CSV",
            df.to_csv(index=False),
            file_name="attendance.csv",
            mime="text/csv",
        )
    else:
        st.info("Belum ada data absensi.")