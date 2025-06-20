import streamlit as st
import pandas as pd
import joblib

# ✅ Set konfigurasi halaman (judul tab browser, ikon, dll)
st.set_page_config(
    page_title="Prediksi Obesitas",
    page_icon="🧠",
    layout="centered"
)

# Load model dan scaler
model = joblib.load("best_random_forest.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Prediksi Tingkat Obesitas 🧠")
st.write("Upload data Anda dan dapatkan prediksi kategori obesitas berdasarkan input yang diberikan.")

# Upload file CSV
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("Data Input:")
        st.dataframe(data)

        # Drop kolom target jika ada
        if 'NObeyesdad' in data.columns:
            data_features = data.drop(columns=['NObeyesdad'])
        else:
            data_features = data

        st.write("Fitur input saat ini:", list(data_features.columns))
        if hasattr(model, 'feature_names_in_'):
            st.write("Fitur saat training:", list(model.feature_names_in_))

        # Normalisasi & prediksi
        data_scaled = scaler.transform(data_features)
        predictions = model.predict(data_scaled)
        data['Prediksi Obesitas'] = predictions

        st.subheader("Hasil Prediksi:")
        st.dataframe(data)

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Hasil Prediksi",
            data=csv,
            file_name='hasil_prediksi_obesitas.csv',
            mime='text/csv',
        )
    except Exception as e:
        st.error(f"Terjadi error saat transform/predict: {e}")
else:
    st.info("Silakan upload file CSV untuk mulai prediksi.")
