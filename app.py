
import streamlit as st
import pandas as pd
import pickle

# Load model, scaler, encoder
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

st.title("Prediksi Kerentanan Ekonomi Nelayan - Kota Kendari")

# Form input
with st.form("input_form"):
    st.subheader("Masukkan Data Nelayan")
    
    jenis_ikan = st.selectbox("Jenis Ikan Utama", ['Tuna', 'Cakalang', 'Tongkol', 'Lainnya'])
    usia = st.number_input("Usia", min_value=18, max_value=80, value=35)
    tanggungan = st.number_input("Jumlah Tanggungan", min_value=0, max_value=10, value=2)
    
    mangrove = st.text_input("Mangrove Terdegradasi", "0.5")
    akses_market = st.text_input("Akses Market", "0.7")
    pencemaran = st.text_input("Indeks Pencemaran", "0.6")
    reklamasi = st.text_input("Indeks Reklamasi", "0.4")
    
    submitted = st.form_submit_button("Prediksi")

# Prediksi jika tombol diklik
if submitted:
    try:
        input_df = pd.DataFrame([[
            encoder.transform([jenis_ikan])[0],
            usia,
            tanggungan,
            float(mangrove.replace(",", ".")),
            float(akses_market.replace(",", ".")),
            float(pencemaran.replace(",", ".")),
            float(reklamasi.replace(",", "."))
        ]], columns=[
            'Jenis_Ikan_Utama', 'Usia', 'Jumlah_Tanggungan',
            'Mangrove_Terdegradasi', 'Akses_Market',
            'Indeks_Pencemaran', 'Indeks_Reklamasi'
        ])
        
        input_scaled = scaler.transform(input_df)
        hasil = model.predict(input_scaled)[0]
        
        st.success(f"Hasil Prediksi: {'Rentan' if hasil == 1 else 'Tidak Rentan'}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses input: {e}")
