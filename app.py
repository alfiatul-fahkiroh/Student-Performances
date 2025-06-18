# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model
with open("model_graduation.pkl", "rb") as file:
    model = joblib.load(file)

# Judul Aplikasi
st.title("Prediksi Kelulusan Mahasiswa (On Time / Late)")
st.write("Masukkan data berikut untuk memprediksi apakah mahasiswa akan lulus tepat waktu atau terlambat.")

# Input pengguna
new_ACT = st.number_input("Nilai ACT composite score", min_value=0.0, max_value=36.0, step=0.1)
new_SAT = st.number_input("Nilai SAT total score", min_value=400.0, max_value=1600.0, step=10.0)
new_GPA = st.number_input("Nilai rata-rata SMA (GPA)", min_value=0.0, max_value=4.0, step=0.01)
new_income = st.number_input("Pendapatan orang tua (dalam USD)", min_value=0.0, step=100.0)
new_education = st.number_input("Tingkat pendidikan orang tua (dalam angka)", min_value=0.0, step=1.0)

# Prediksi
if st.button("Prediksi"):
    try:
        # Dataframe dari input
        new_data_df = pd.DataFrame(
            [[new_ACT, new_SAT, new_GPA, new_income, new_education]],
            columns=['ACT composite score', 'SAT total score', 'high school gpa', 'parental income', 'parent_edu_numerical']
        )

        # Prediksi
        predicted_code = model.predict(new_data_df)[0]
        label_mapping = {1: 'On Time', 0: 'Late'}
        predicted_label = label_mapping.get(predicted_code, 'Tidak diketahui')

        st.success(f"Prediksi kategori masa studi: **{predicted_label}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
