"""
📸 Sistem Absen Wajah Siswa MEC (versi perbaikan)

Perbaikan utama dari versi sebelumnya:
- Pendaftaran wajah (ambil foto + hitung embedding) HANYA dilakukan SEKALI
  per siswa lewat menu "Daftar Wajah Baru".
- Menu "Absen Wajah" TIDAK meminta data lagi - cukup ambil foto, sistem
  otomatis mengenali siapa siswanya dari data yang sudah tersimpan lalu
  langsung mencatat ke Google Sheets.
- Menyimpan embedding secara lokal (data/embeddings.pkl) sehingga tidak perlu
  menghubungi API eksternal atau menghitung ulang wajah tiap kali absen.
- Mencegah absen ganda: satu siswa hanya tercatat sekali per hari.
"""

import os
import tempfile
import streamlit as st
import pandas as pd

from utils import face_recognition as fr
from utils import sheets as sh

st.set_page_config(page_title="Absen Wajah MEC", page_icon="📸", layout="centered")

MENU = ["🏠 Beranda", "📝 Daftar Wajah Baru", "📷 Absen Wajah", "📊 Lihat Rekap", "⚙️ Kelola Data Wajah"]


def simpan_upload_sementara(uploaded_file_or_camera) -> str:
    """Simpan foto (dari kamera atau upload) ke file sementara agar bisa diproses DeepFace."""
    suffix = ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file_or_camera.getbuffer())
    tmp.close()
    return tmp.name


def halaman_beranda():
    st.title("📸 Sistem Absen Wajah Siswa MEC")
    st.write(
        "Sistem absensi otomatis berbasis pengenalan wajah. "
        "**Daftarkan wajah siswa cukup sekali**, setelah itu siswa cukup "
        "melakukan absen dengan foto — sistem akan mengenali wajah secara otomatis."
    )
    st.markdown(
        """
        ### Alur pemakaian
        1. **Daftar Wajah Baru** (dilakukan sekali per siswa, oleh admin/guru)
        2. **Absen Wajah** (dilakukan siswa setiap hari, tinggal foto)
        3. **Lihat Rekap** (melihat/mengunduh data absensi)
        """
    )
    terdaftar = fr.list_registered()
    st.info(f"Jumlah siswa yang sudah terdaftar: **{len(terdaftar)}**")


def halaman_daftar_wajah():
    st.title("📝 Daftar Wajah Baru")
    st.caption("Dilakukan SEKALI saja per siswa. Setelah ini siswa tinggal absen tanpa perlu daftar ulang.")

    with st.form("form_daftar", clear_on_submit=True):
        nis = st.text_input("NIS / ID Siswa")
        nama = st.text_input("Nama Lengkap")
        kelas = st.text_input("Kelas")
        sumber_foto = st.radio("Ambil foto dari", ["Kamera", "Upload File"], horizontal=True)

        foto = None
        if sumber_foto == "Kamera":
            foto = st.camera_input("Ambil foto wajah")
        else:
            foto = st.file_uploader("Upload foto wajah", type=["jpg", "jpeg", "png"])

        submit = st.form_submit_button("Daftarkan Wajah")

    if submit:
        if not nis or not nama or not kelas:
            st.error("Mohon lengkapi NIS, Nama, dan Kelas.")
        elif foto is None:
            st.error("Mohon ambil atau upload foto terlebih dahulu.")
        else:
            with st.spinner("Memproses wajah..."):
                path = simpan_upload_sementara(foto)
                try:
                    pesan = fr.register_face(nis.strip(), nama.strip(), kelas.strip(), path)
                    st.success(pesan)
                except ValueError:
                    st.error("Wajah tidak terdeteksi pada foto. Coba foto dengan pencahayaan lebih baik.")
                finally:
                    os.remove(path)


def halaman_absen():
    st.title("📷 Absen Wajah")
    st.caption("Cukup ambil foto — sistem otomatis mengenali wajah dari data yang sudah terdaftar.")

    foto = st.camera_input("Ambil foto untuk absen")

    if foto is not None:
        with st.spinner("Mengenali wajah..."):
            path = simpan_upload_sementara(foto)
            try:
                nis, nama, kelas, jarak = fr.recognize_face(path)
            except ValueError:
                nis = None
            finally:
                os.remove(path)

        if nis is None:
            st.error("❌ Wajah tidak dikenali. Pastikan sudah terdaftar di menu 'Daftar Wajah Baru', "
                      "atau coba dengan pencahayaan/posisi wajah yang lebih baik.")
        else:
            st.success(f"Wajah dikenali: **{nama}** ({kelas}) — NIS {nis}")
            with st.spinner("Mencatat absensi ke Google Sheets..."):
                pesan = sh.catat_absen(nis, nama, kelas)
            st.info(pesan)


def halaman_rekap():
    st.title("📊 Lihat Rekap Absensi")

    with st.spinner("Mengambil data dari Google Sheets..."):
        try:
            data = sh.ambil_rekap()
        except Exception as e:
            st.error(f"Gagal mengambil data dari Google Sheets: {e}")
            return

    if not data:
        st.warning("Belum ada data absensi.")
        return

    df = pd.DataFrame(data)

    col1, col2 = st.columns(2)
    with col1:
        tanggal_filter = st.text_input("Filter tanggal (YYYY-MM-DD, kosongkan untuk semua)")
    with col2:
        nama_filter = st.text_input("Filter nama (kosongkan untuk semua)")

    if tanggal_filter:
        df = df[df["Tanggal"] == tanggal_filter]
    if nama_filter:
        df = df[df["Nama"].str.contains(nama_filter, case=False, na=False)]

    st.dataframe(df, use_container_width=True)
    st.download_button(
        "⬇️ Unduh sebagai CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="rekap_absensi.csv",
        mime="text/csv",
    )


def halaman_kelola():
    st.title("⚙️ Kelola Data Wajah Terdaftar")
    terdaftar = fr.list_registered()

    if not terdaftar:
        st.warning("Belum ada wajah yang terdaftar.")
        return

    df = pd.DataFrame(terdaftar)
    st.dataframe(df, use_container_width=True)

    nis_hapus = st.selectbox("Pilih NIS untuk dihapus", [d["nis"] for d in terdaftar])
    if st.button("🗑️ Hapus Data Ini", type="secondary"):
        pesan = fr.delete_face(nis_hapus)
        st.success(pesan)
        st.rerun()


def main():
    st.sidebar.title("📸 Menu")
    pilihan = st.sidebar.radio("Navigasi", MENU)

    if pilihan == "🏠 Beranda":
        halaman_beranda()
    elif pilihan == "📝 Daftar Wajah Baru":
        halaman_daftar_wajah()
    elif pilihan == "📷 Absen Wajah":
        halaman_absen()
    elif pilihan == "📊 Lihat Rekap":
        halaman_rekap()
    elif pilihan == "⚙️ Kelola Data Wajah":
        halaman_kelola()


if __name__ == "__main__":
    main()