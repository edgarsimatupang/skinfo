import streamlit as st
from transformers import pipeline

def load_model():
    return pipeline(
        "text-classification",
        model="tabularisai/multilingual-sentiment-analysis"
    )

sentiment_pipeline = load_model()

st.set_page_config(page_title="Multilingual Sentiment Analysis", page_icon="🌍")
st.title("Multilingual Sentiment Analysis")
st.write("Model by `tabularisai/multilingual-sentiment-analysis`")

user_input = st.text_area("Masukkan teks untuk dianalisis:")

if st.button("Analisis Sentimen"):
    if user_input.strip() == "":
        st.warning("Tolong masukkan teks terlebih dahulu.")
    else:
        with st.spinner("Sedang menganalisis..."):
            try:
                results = sentiment_pipeline(user_input)
                for idx, res in enumerate(results):
                    label = res['label']
                    score = res['score']
                    st.success(f"**Hasil {idx+1}:** {label} (confidence: {score:.2f})")
            except Exception as e:
                st.error(f"Terjadi error: {e}")
