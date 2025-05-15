import streamlit as st
from transformers import pipeline

def load_model():
    return pipeline(
        "sentiment-analysis",  
        model="tabularisai/multilingual-sentiment-analysis",
        device=-1 
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
                if not isinstance(results, list):
                    results = [results]
                for idx, res in enumerate(results):
                    label = res.get('label', 'Unknown')
                    score = res.get('score', 0)
                    st.success(f"**Hasil {idx+1}:** {label} (confidence: {score:.2f})")
            except Exception as e:
                st.error(f"Terjadi error: {e}")
