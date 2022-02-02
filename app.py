import app_validasi
import app_prediksi
import streamlit as st

PAGES = {
  "Validasi": app_validasi,
  "Prediksi": app_prediksi,
}

st.set_page_config(page_title="Prediksi dan Validasi Model LSTM Debit Air")

st.sidebar.title("Navigasi Aplikasi")
selection = st.sidebar.radio("Aplikasi", list(PAGES.keys()))

page = PAGES[selection]

page.app()