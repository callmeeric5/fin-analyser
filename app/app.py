import streamlit as st
from src.data_preprocess import preprocess
from src.ocr import OCR
from src.llm import call_qwen_api, call_qdrant_api
from src.display import display_Qwen
from PIL import Image
import tempfile
import os

st.title("Financial Assistant")
st.write("👋 Use your personal financial assistant to undertand your spends well!")
st.write("Upload your first invoice, bank statement,receipt, etc to start 👇")

model_helper = st.selectbox(
    "pick a LLM for chatting", ["Qwen", "Gemini", "LLama", "DeepSeek"]
)
file = st.file_uploader("upload file here")


@st.cache_resource(show_spinner=False)
def get_ocr():
    return OCR(languages=["en"], gpu=False)


ocr_engine = get_ocr()

if file:
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(file.name)[1]
    ) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    extracted_data = []
    output = preprocess(file)
    images = output["img"]
    page_text = ""

    if output["text"]:
        st.subheader("Raw text detected")
        st.write(output["text"])
        page_text += output["text"]

    if images:
        if isinstance(images, Image.Image):  # single image → wrap
            images = [images]

        st.subheader("OCR on image pages")
        for idx, img in enumerate(images, start=1):
            ocr_results = ocr_engine.extract(img)  # list[dict]
            annotated = ocr_engine.annotate(img, results=ocr_results, box_thickness=2)

            with st.expander(f"Page / Image {idx}"):
                st.image(annotated, caption="OCR annotations", use_container_width=True)

    if st.button("Process with LLM"):
        st.subheader("LLM Output")
        with st.spinner("Processing with LLM..."):
            try:
                if model_helper == "Qwen":
                    llm_output = call_qwen_api(tmp_file_path)
                    display_Qwen(llm_output)
                else:
                    llm_output = call_qdrant_api(page_text, model_helper)
            finally:
                # Clean up the temporary file
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
