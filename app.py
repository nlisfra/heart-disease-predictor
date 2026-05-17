import streamlit as st
import joblib
import numpy as np

# ── Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="centered"
)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ── UI ────────────────────────────────────────────────────────
st.title("🫀 Heart Disease Predictor")
st.write("Masukkan data kesehatan pasien untuk memprediksi risiko penyakit jantung.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    age      = st.number_input("Umur (tahun)", min_value=1, max_value=120, value=50)
    sex      = st.selectbox("Jenis Kelamin", options=[1, 0], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
    cp       = st.selectbox("Tipe Nyeri Dada (cp)", options=[0, 1, 2, 3],
                             format_func=lambda x: {
                                 0: "0 - Typical Angina",
                                 1: "1 - Atypical Angina",
                                 2: "2 - Non-anginal Pain",
                                 3: "3 - Asymptomatic"
                             }[x])
    trestbps = st.number_input("Tekanan Darah Istirahat (mm Hg)", min_value=80, max_value=220, value=120)
    chol     = st.number_input("Kolesterol Serum (mg/dl)", min_value=100, max_value=600, value=200)
    fbs      = st.selectbox("Gula Darah Puasa > 120 mg/dl (fbs)",
                             options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    restecg  = st.selectbox("Hasil EKG Istirahat (restecg)", options=[0, 1, 2],
                              format_func=lambda x: {
                                  0: "0 - Normal",
                                  1: "1 - Abnormalitas ST-T",
                                  2: "2 - Hipertrofi Ventrikel Kiri"
                              }[x])

with col2:
    thalach  = st.number_input("Detak Jantung Maksimum (thalach)", min_value=60, max_value=220, value=150)
    exang    = st.selectbox("Angina akibat Olahraga (exang)",
                             options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    oldpeak  = st.slider("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
    slope    = st.selectbox("Slope ST Segment", options=[0, 1, 2],
                             format_func=lambda x: {
                                 0: "0 - Upsloping",
                                 1: "1 - Flat",
                                 2: "2 - Downsloping"
                             }[x])
    ca       = st.selectbox("Jumlah Pembuluh Darah Utama (ca)", options=[0, 1, 2, 3])
    thal     = st.selectbox("Thalassemia (thal)", options=[1, 2, 3],
                             format_func=lambda x: {
                                 1: "1 - Fixed Defect",
                                 2: "2 - Normal",
                                 3: "3 - Reversable Defect"
                             }[x])

st.divider()

# ── Prediksi ──────────────────────────────────────────────────
if st.button("🔍 Prediksi Sekarang", use_container_width=True, type="primary"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    result       = model.predict(input_scaled)[0]
    prob         = model.predict_proba(input_scaled)[0]

    st.subheader("Hasil Prediksi")

    if result == 1:
        st.error(f"⚠️ **Terdeteksi Risiko Penyakit Jantung**")
        st.metric("Probabilitas Berisiko", f"{prob[1]:.1%}")
        st.warning("Disarankan untuk segera berkonsultasi dengan dokter spesialis jantung.")
    else:
        st.success(f"✅ **Risiko Penyakit Jantung Rendah**")
        st.metric("Probabilitas Aman", f"{prob[0]:.1%}")
        st.info("Tetap jaga pola hidup sehat dan lakukan pemeriksaan rutin.")

    # Detail probabilitas
    with st.expander("📊 Detail Probabilitas"):
        st.write(f"- Tidak berisiko: **{prob[0]:.1%}**")
        st.write(f"- Berisiko: **{prob[1]:.1%}**")

st.divider()
st.caption("⚠️ Aplikasi ini hanya untuk tujuan edukasi dan portofolio. Bukan pengganti diagnosis medis profesional.")
