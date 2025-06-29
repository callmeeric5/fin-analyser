import streamlit as st
from src.data_preprocess import preprocess

st.title("Financial Assistant")
st.write("👋 Use your personal financial assistant to undertand your spends well!")
st.write("Upload your first invoice, bank statement,receipt, etc to start 👇")

ocr_helper = st.selectbox("pick an ocr tool", ["EasyOCR", "YOLO"])
model_helper = st.selectbox(
    "pick a LLM for chatting", ["Gemini", "LLama", "DeepSeek", "Qwen"]
)
files = st.file_uploader("upload file here", accept_multiple_files=True)

if files:
    extracted_data = []
    for file in files:
        output = preprocess(file)
        st.write(output)
        st.write(output["text"])
        st.write(output["img"])

# TODO OCR Parse
# TODO RAG
# TODO LLM
