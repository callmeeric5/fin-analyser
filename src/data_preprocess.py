from PyPDF2 import PdfReader
from sympy import im
from pdf2image import convert_from_bytes
from PIL import Image
import pandas as pd
import streamlit as st
from io import BytesIO


def preprocess(file):
    byte_input = file.read()
    output = {"text": "", "img": []}
    file_suffix = file.name.split(".")[-1].lower()
    match file_suffix:
        case "pdf":
            reader = PdfReader(BytesIO(byte_input))
            for page in reader.pages:
                output["text"] += page.extract_text() or ""
            imges = convert_from_bytes(byte_input,dpi=150,thread_count=4)
            output["img"] = imges
        case "csv":
            df = pd.read_csv(BytesIO(byte_input))
            output["text"] = df.to_string()
        case "txt" | "md":
            output["text"] = byte_input.decode("utf-8")
        case "png" | "jpg" | "jpeg":
            output["img"] = Image.open(BytesIO(byte_input))

        case _:
            output["text"] = "unsupported file type"
    return output
