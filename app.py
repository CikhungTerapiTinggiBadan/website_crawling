import streamlit as st
import pandas as pd
import joblib
import numpy as np

#Load model
@st.cache_resource
def load_model():
    return joblib.load("model_prediksi_viralitas.pkl")

model = load_model()

#Setup tampilan
st.set_page_config(
    page_title="Prediksi Viralitas Lagu 🎵",
    page_icon="🎶",
    layout="centered"
)

st.title("🎵 Prediksi Viralitas Lagu YouTube 2025")
st.write("Masukkan judul lagu untuk mengetahui seberapa besar kemungkinan lagu tersebut akan **viral** di YouTube!")

#Input dari pengguna
judul = st.text_input("Judul Lagu", placeholder="Contoh: My Love Story (Official Music Video)")

#Prediksi probabilitas
if st.button("Prediksi Viralitas"):
    if judul.strip() == "":
        st.warning("⚠️ Masukkan judul lagu terlebih dahulu.")
    else:
        #Gabungkan semua teks
        teks_input = f"{judul}"
        prob_viral = model.predict_proba([teks_input])[0][1]
        pred_label = "Viral" if prob_viral > 0.5 else "Tidak Viral"

        #Tampilkan hasil
        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")
        st.metric(
            label="Kemungkinan Viral (%)",
            value=f"{prob_viral * 100:.2f}%",
            delta="🟢 Viral" if pred_label == "Viral" else "🔴 Tidak Viral"
        )

        #Feedback visual
        if prob_viral > 0.75:
            st.success("🔥 Lagu ini berpotensi besar menjadi viral!")
        elif prob_viral > 0.5:
            st.info("💡 Lagu ini punya peluang cukup besar untuk viral.")
        else:
            st.warning("😐 Lagu ini kemungkinan tidak terlalu viral.")