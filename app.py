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

# Form input
with st.form("form_nelayan"):
    produksi = st.number_input("Produksi Tahunan (kg)", min_value=0)
    jenis_ikan = st.selectbox("Jenis Ikan Utama", label_encoders['Jenis_Ikan_Utama'].classes_)
    pendapatan = st.number_input("Pendapatan Rata-rata per Tahun (Rp)", min_value=0)
    alat_tangkap = st.selectbox("Jenis Alat Tangkap", label_encoders['Jenis_Alat_Tangkap'].classes_)
    lama_usaha = st.number_input("Lama Berpenghasilan (tahun)", min_value=0)
    mangrove = st.slider("Tingkat Kerusakan Mangrove (%)", 0.0, 1.0, step=0.01)
    akses_market = st.number_input("Jarak ke Akses Market (km)", min_value=0.0)
    pencemaran = st.slider("Indeks Pencemaran (0 - 1)", 0.0, 1.0, step=0.01)
    reklamasi = st.slider("Indeks Reklamasi (0 - 1)", 0.0, 1.0, step=0.01)
    bantuan = st.radio("Pernah Dapat Bantuan?", ["ya", "tidak"])
    pendidikan = st.selectbox("Pendidikan Terakhir", label_encoders['Pendidikan_Terakhir'].classes_)

    submit = st.form_submit_button("Prediksi")

# Proses prediksi
if submit:
    data_input = pd.DataFrame([{
        'produksi_tahunan': produksi,
        'Jenis_Ikan_Utama': jenis_ikan,
        'Pendapatan_Rata2': pendapatan,
        'Jenis_Alat_Tangkap': alat_tangkap,
        'Lama_Berpenghasilan': lama_usaha,
        'Mangrove_Terdegradasi': mangrove,
        'Akses_Market': akses_market,
        'Indeks_Pencemaran': pencemaran,
        'Indeks_Reklamasi': reklamasi,
        'Ada_Bantuan': 1 if bantuan == 'ya' else 0,
        'Pendidikan_Terakhir': pendidikan
    }])

    # Cek kolom
st.write("Kolom dari input:", data_input.columns.tolist())
st.write("Kolom dari label encoder:", list(label_encoders.keys()))

# Transformasi hanya jika kolom cocok
for col in label_encoders:
    if col in data_input.columns:
        data_input[col] = label_encoders[col].transform(data_input[col])
    else:
        st.warning(f"⚠️ Kolom '{col}' tidak ditemukan. Cek lagi nama kolom input.")
        

    for col in label_encoders:
        data_input[col] = label_encoders[col].transform(data_input[col])

    data_scaled = scaler.transform(data_input)
    pred = model.predict(data_scaled)[0]
    label = target_encoder.inverse_transform([pred])[0]

    st.success(f"Kategori Kerentanan Ekonomi: {label}")
