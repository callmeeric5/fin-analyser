import streamlit as st
from src.data_preprocess import preprocess
from src.ocr import OCR
from PIL import Image


st.title("Financial Assistant")
st.write("👋 Use your personal financial assistant to undertand your spends well!")
st.write("Upload your first invoice, bank statement,receipt, etc to start 👇")

model_helper = st.selectbox(
    "pick a LLM for chatting", ["Gemini", "LLama", "DeepSeek", "Qwen"]
)
file = st.file_uploader("upload file here")


@st.cache_resource(show_spinner=False)
def get_ocr():
    return OCR(languages=["en"], gpu=False)


ocr_engine = get_ocr()

if file:
    extracted_data = []
    output = preprocess(file)
    images = output["img"]
    if output["text"]:
        st.subheader("Raw text detected")
        st.write(output["text"])
    if images:
        if isinstance(images, Image.Image):  # single image → wrap
            images = [images]

        st.subheader("OCR on image pages")
        for idx, img in enumerate(images, start=1):
            ocr_results = ocr_engine.extract(img)  # list[dict]

            page_text = " ".join(r["text"] for r in ocr_results)  # type: ignore

            annotated = ocr_engine.annotate(img, results=ocr_results)

            with st.expander(f"Page / Image {idx}"):
                st.image(annotated, caption="OCR annotations", use_container_width=True)
                st.write("### Detected text")
                st.write(page_text)  # or st.text(page_text)

# TODO OCR Parse
# TODO RAG
# TODO LLM
