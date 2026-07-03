import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
from deepface import DeepFace
from streamlit_gsheets import GSheetsConnection

# Konfigurasi halaman
st.set_page_config(
    page_title="Absen Anak Kampung",
    page_icon="📸",
    layout="centered"
)

# Title
st.title("📸 Sistem Absen Wajah")
st.markdown("### Untuk Anak-Anak Kampung")

# Inisialisasi koneksi Google Sheets
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
    USE_GSHEETS = True
except Exception as e:
    st.warning("⚠️ Tidak terhubung ke Google Sheets. Data akan disimpan di CSV lokal.")
    USE_GSHEETS = False

# Setup folder untuk data
FACE_DB = "face_database"
ATTENDANCE_DIR = "attendance"
os.makedirs(FACE_DB, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# ========== FUNGSI UTAMA ==========

def register_face(name, image):
    """Daftarkan wajah baru ke database"""
    try:
        # Simpan foto ke folder database
        filename = os.path.join(FACE_DB, f"{name}.jpg")
        cv2.imwrite(filename, image)
        return True
    except Exception as e:
        st.error(f"Gagal mendaftarkan wajah: {e}")
        return False

def detect_face_from_image(image):
    """Deteksi wajah dari gambar yang diambil/upload"""
    try:
        # Simpan sementara gambar untuk diproses
        temp_path = "temp_face.jpg"
        cv2.imwrite(temp_path, image)
        
        # Cari wajah di database
        result = DeepFace.find(
            img_path=temp_path, 
            db_path=FACE_DB,
            enforce_detection=False,
            model_name='Facenet'
        )
        
        # Hapus file temporary
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if result and not result[0].empty:
            # Ambil nama dari path file
            nama = result[0]['identity'][0].split('/')[-1].split('.')[0]
            return nama
        else:
            return None
    except Exception as e:
        st.error(f"Error saat deteksi: {e}")
        return None

def save_attendance_to_gsheets(name, waktu, tanggal):
    """Simpan ke Google Sheets"""
    try:
        # Buat data baru
        new_data = pd.DataFrame([[name, waktu, tanggal]], columns=["Nama", "Waktu", "Tanggal"])
        
        # Coba baca data yang sudah ada
        existing_df = conn.read(worksheet="Sheet1")
        
        # Gabungkan
        updated_df = pd.concat([existing_df, new_data], ignore_index=True)
        
        # Tulis ke Google Sheets
        conn.write(data=updated_df, worksheet="Sheet1")
        return True
    except Exception as e:
        st.error(f"Gagal simpan ke Google Sheets: {e}")
        return False

def save_attendance_to_csv(name, waktu, tanggal):
    """Simpan ke CSV lokal (cadangan)"""
    filename = os.path.join(ATTENDANCE_DIR, f"absensi_{tanggal}.csv")
    
    new_data = pd.DataFrame([[name, waktu, tanggal]], columns=["Nama", "Waktu", "Tanggal"])
    
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        if name in df['Nama'].values:
            st.warning(f"⚠️ {name} sudah absen hari ini!")
            return False
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_csv(filename, index=False)
    else:
        new_data.to_csv(filename, index=False)
    
    return True

def save_attendance(name):
    """Fungsi utama untuk menyimpan absensi"""
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")
    
    # Cek apakah sudah absen hari ini di Google Sheets
    if USE_GSHEETS:
        try:
            existing_df = conn.read(worksheet="Sheet1")
            if name in existing_df['Nama'].values:
                st.warning(f"⚠️ {name} sudah absen hari ini!")
                return
        except:
            pass  # Jika sheet kosong, lanjutkan
    
    # Simpan ke Google Sheets
    if USE_GSHEETS:
        success = save_attendance_to_gsheets(name, now, today)
        if success:
            st.success(f"✅ {name} berhasil absen pada {now} (tersimpan di Google Sheets)")
        else:
            # Fallback ke CSV jika GSheets gagal
            success_csv = save_attendance_to_csv(name, now, today)
            if success_csv:
                st.success(f"✅ {name} berhasil absen pada {now} (tersimpan di CSV lokal)")
    else:
        # Simpan ke CSV
        success_csv = save_attendance_to_csv(name, now, today)
        if success_csv:
            st.success(f"✅ {name} berhasil absen pada {now} (tersimpan di CSV lokal)")

# ========== MENU UTAMA ==========

menu = st.sidebar.radio(
    "Pilih Menu:",
    ["📸 Absen Wajah", "📝 Daftar Wajah Baru", "📊 Lihat Rekap"]
)

# ========== MENU 1: ABSEN WAJAH ==========
if menu == "📸 Absen Wajah":
    st.subheader("📸 Ambil Foto untuk Absen")
    
    # Pilihan metode
    metode = st.radio(
        "Pilih cara absen:",
        ["📷 Foto Langsung (Rekomendasi)", "📤 Upload Foto"]
    )
    
    if metode == "📷 Foto Langsung (Rekomendasi)":
        foto = st.camera_input("Klik tombol untuk ambil foto")
        
        if foto is not None:
            # Konversi ke OpenCV
            bytes_data = foto.getvalue()
            nparr = np.frombuffer(bytes_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Tampilkan preview
            st.image(img, channels="BGR", caption="Foto yang diambil", width=300)
            
            # Proses deteksi
            with st.spinner("🔍 Mencocokkan wajah..."):
                nama = detect_face_from_image(img)
                
                if nama:
                    save_attendance(nama)
                else:
                    st.error("❌ Wajah tidak dikenali. Pastikan kamu sudah terdaftar!")
                    
                    # Tampilkan info jika belum terdaftar
                    with st.expander("ℹ️ Belum terdaftar?"):
                        st.write("Silakan ke menu **📝 Daftar Wajah Baru** untuk mendaftar.")
    
    else:  # Upload foto
        uploaded_file = st.file_uploader("Upload foto wajah", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            nparr = np.frombuffer(bytes_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            st.image(img, channels="BGR", caption="Foto yang diupload", width=300)
            
            with st.spinner("🔍 Mencocokkan wajah..."):
                nama = detect_face_from_image(img)
                
                if nama:
                    save_attendance(nama)
                else:
                    st.error("❌ Wajah tidak dikenali. Pastikan kamu sudah terdaftar!")

# ========== MENU 2: DAFTAR WAJAH BARU ==========
elif menu == "📝 Daftar Wajah Baru":
    st.subheader("📝 Daftarkan Wajah Baru")
    
    nama = st.text_input("Masukkan Nama Lengkap:")
    
    if nama:
        st.write("📸 Ambil foto wajah (pastikan wajah terlihat jelas)")
        foto = st.camera_input("Klik tombol untuk mengambil foto")
        
        if foto is not None:
            bytes_data = foto.getvalue()
            nparr = np.frombuffer(bytes_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            st.image(img, channels="BGR", caption="Foto yang diambil", width=300)
            
            if st.button("✅ Daftarkan Wajah"):
                with st.spinner("Mendaftarkan wajah..."):
                    success = register_face(nama, img)
                    if success:
                        st.success(f"✅ {nama} berhasil didaftarkan!")
                        st.info("Sekarang kamu bisa absen menggunakan wajah.")
                    else:
                        st.error("❌ Gagal mendaftarkan wajah. Coba lagi.")

# ========== MENU 3: LIHAT REKAP ==========
else:  # Lihat Rekap
    st.subheader("📊 Rekap Absensi")
    
    if USE_GSHEETS:
        try:
            df = conn.read(worksheet="Sheet1")
            
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                st.metric("Total Absensi", len(df))
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"absensi_{datetime.now().strftime('%Y-%m-%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("📭 Belum ada data absensi")
        except:
            st.info("📭 Belum ada data absensi di Google Sheets")
    else:
        # Tampilkan dari CSV lokal
        today = datetime.now().strftime("%Y-%m-%d")
        filepath = os.path.join(ATTENDANCE_DIR, f"absensi_{today}.csv")
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            st.dataframe(df, use_container_width=True)
            st.metric("Total Absensi Hari Ini", len(df))
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"absensi_{today}.csv",
                mime="text/csv"
            )
        else:
            st.info(f"📭 Belum ada absensi hari ini ({today})")

# ========== FOOTER ==========
st.sidebar.markdown("---")
st.sidebar.info("💡 Pastikan wajah terlihat jelas dan pencahayaan cukup")
st.sidebar.caption("Sistem Absen Anak Kampung v1.0")