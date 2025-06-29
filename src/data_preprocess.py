from PyPDF2 import PdfReader
from sympy import im
from pdf2image import convert_from_bytes
from PIL import Image
import pandas as pd
import streamlit as st


def preprocess(file):
    output = {"text": "", "img": []}
    file_suffix = file.name.split(".")[-1].lower()
    print(file_suffix)
    if file_suffix == "pdf":
        print("start")
        reader = PdfReader(file)
        for page in reader.pages:
            output["text"] += page.extract_text()

        imges = convert_from_bytes(file.read())
        output["img"] = imges

    return output
