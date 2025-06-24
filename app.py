import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model dan alat bantu
model = joblib.load('ada_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoder.pkl')
target_encoder = joblib.load('target_encoder.pkl')

st.title("📊 Prediksi Kerentanan Ekonomi Nelayan")
st.markdown("Masukkan data nelayan untuk melihat kategori kerentanannya.")

with st.form("form_nelayan"):
    # Input kategori
    jenis_ikan = st.selectbox("Jenis Ikan Utama", label_encoders['Jenis_Ikan_Utama'].classes_)
    alat_tangkap = st.selectbox("Jenis Alat Tangkap", label_encoders['Jenis_Alat_Tangkap'].classes_)
    pendidikan = st.selectbox("Pendidikan Terakhir", label_encoders['Pendidikan_Terakhir'].classes_)

    # Input numerik
    produksi = st.number_input("Produksi Tahunan (kg)", min_value=0)
    pendapatan = st.number_input("Pendapatan Rata-rata per Tahun (Rp)", min_value=0)
    lama_usaha = st.number_input("Lama Berpenghasilan (tahun)", min_value=0)
    mangrove = st.slider("Tingkat Kerusakan Mangrove (%)", 0.0, 1.0, step=0.01)
    akses_market = st.number_input("Jarak ke Akses Market (km)", min_value=0.0)
    pencemaran = st.slider("Indeks Pencemaran (0 - 1)", 0.0, 1.0, step=0.01)
    reklamasi = st.slider("Indeks Reklamasi (0 - 1)", 0.0, 1.0, step=0.01)

    submit = st.form_submit_button("Prediksi")

if submit:
    # Encode kategori
    jenis_ikan_encoded = label_encoders['Jenis_Ikan_Utama'].transform([jenis_ikan])[0]
    alat_tangkap_encoded = label_encoders['Jenis_Alat_Tangkap'].transform([alat_tangkap])[0]
    pendidikan_encoded = label_encoders['Pendidikan_Terakhir'].transform([pendidikan])[0]

    # Gabungkan jadi satu array
    fitur_model = pd.DataFrame([{
        'produksi_tahunan': produksi,
        'Pendapatan_Rata2': pendapatan,
        'Lama_Berpenghasilan': lama_usaha,
        'Mangrove_Terdegradasi': mangrove,
        'Akses_Market': akses_market,
        'Indeks_Pencemaran': pencemaran,
        'Indeks_Reklamasi': reklamasi
    }])

    # Scaler & prediksi
    data_scaled = scaler.transform(fitur_model)
    pred = model.predict(data_scaled)[0]
    hasil = target_encoder.inverse_transform([pred])[0]

    st.success(f"✅ Kategori Kerentanan Ekonomi: {hasil}")
