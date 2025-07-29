import pandas as pd
import streamlit as st
from src.data_preprocess import preprocess
from src.ocr import OCR
from src.llm import call_qwen_api, call_groq_api, call_gemini_api
from src.display import display
from PIL import Image
import tempfile
import os
import plotly.express as px

PARQUET_PATH = "data/data.parquet"
st.set_page_config(page_title="Financial Analyzer")
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Visualization"])


@st.cache_resource(show_spinner=False)
def get_ocr():
    return OCR(languages=["en"], gpu=False)


ocr_engine = get_ocr()
if page == "Home":
    st.title("Financial Assistant")
    st.write("👋 Use your personal financial assistant to undertand your spends well!")
    st.write("Upload your first invoice, bank statement,receipt, etc to start 👇")

    model_helper = st.selectbox(
        "pick a LLM for chatting",
        ["Llama", "Gemini", "Qwen"],
    )
    file = st.file_uploader("upload file here")

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
                annotated = ocr_engine.annotate(
                    img, results=ocr_results, box_thickness=2
                )

                with st.expander(f"Page / Image {idx}"):
                    st.image(
                        annotated, caption="OCR annotations", use_container_width=True
                    )
        if st.button("Process with LLM"):
            st.subheader("LLM Output")
            with st.spinner("Processing with LLM..."):
                try:
                    if model_helper == "Qwen":
                        llm_output = call_qwen_api(tmp_file_path)
                    elif model_helper == "Llama":
                        llm_output = call_groq_api(tmp_file_path)
                    elif model_helper == "Gemini":
                        llm_output = call_gemini_api(tmp_file_path)
                    display(llm_output)
                    os.makedirs("data", exist_ok=True)
                    df_new = pd.DataFrame([llm_output])
                    if os.path.exists(PARQUET_PATH):
                        df_existing = pd.read_parquet(PARQUET_PATH)
                        df_all = pd.concat([df_existing, df_new], ignore_index=True)

                    else:
                        df_all = df_new
                    df_all.to_parquet(PARQUET_PATH, index=False)
                    st.success("✅ Data saved!")

                finally:
                    if os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)

elif page == "Visualization":
    st.title("📈 Invoice History Dashboard")

    if not os.path.exists(PARQUET_PATH):
        st.warning("No invoice history found yet.")
        st.stop()

    df = pd.read_parquet(PARQUET_PATH)
    st.caption(f"Loaded {len(df)} records")

    df["total"] = pd.to_numeric(df["total"], errors="coerce")
    df["subtotal"] = pd.to_numeric(df.get("subtotal", 0), errors="coerce")
    df["tax"] = pd.to_numeric(df.get("tax", 0), errors="coerce")
    df["tips"] = pd.to_numeric(df.get("tips", 0), errors="coerce")

    df.dropna(subset=["total"], inplace=True)

    df["combined_date"] = df["date"].astype(str) + df["invoice_date"].astype(str)
    df["month"] = df["combined_date"].astype(str).str.slice(3, 5)

    month_order = [f"{i:02d}" for i in range(1, 13)]
    df["month"] = pd.Categorical(df["month"], categories=month_order, ordered=True)

    st.subheader("📊 Total Spending by Month")
    monthly = df.groupby("month")["total"].sum().reset_index()
    fig1 = px.bar(
        monthly,
        x="month",
        y="total",
        title="Total Spending by Month",
        labels={"month": "Month", "total": "Amount"},
    )
    st.plotly_chart(fig1)

    if "store_name" in df.columns:
        st.subheader("🏬 Spending by Store Name")
        store_spend = df.groupby("store_name")["total"].sum().reset_index()
        fig2 = px.bar(
            store_spend, x="store_name", y="total", title="Total by Store Address"
        )
        st.plotly_chart(fig2)
    st.subheader("💸 Breakdown: Subtotal + Tax + Tips")
    breakdown = df.groupby("month")[["subtotal", "tax", "tips"]].sum().reset_index()
    fig3 = px.bar(
        breakdown,
        x="month",
        y=["subtotal", "tax", "tips"],
        barmode="stack",
        title="Financial Breakdown per Month",
    )
    st.plotly_chart(fig3)

    st.subheader("📂 All Transactions")
    st.dataframe(df.sort_values("date", ascending=False), use_container_width=True)
