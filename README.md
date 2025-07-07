# Finicial Analyzer
## Description
A smart financial management tool that processes multi-format inputs (PDF, CSV, TXT, PNG, JPEG) to extract invoice fields using OCR and LLMmodels. Extracted data is stored in a Vector DB for natural language queries via RAG and visualized for financial insights. Built and validated using a Kaggle invoice dataset.

## Features

- Multi-Format Input: Supports PDF, CSV, TXT, PNG, and JPEG files.

- Field Extraction:
EasyOCR + Google Gemini: High-accuracy OCR with multimodal LLM for field extraction (e.g., client name, invoice number, date).


- Data Storage: Stores extracted fields in FAISS Vector DB for efficient retrieval.

- RAG Q&A: Enables natural language queries (e.g., "What is my total spending?") via Gemini or OpenAI LLM.

- Visualization: Generates financial trends and distributions using Plotly.


## Tech Stack

- OCR: EasyOCR

- LLM: Google Gemini, OpenAI GPT-4

- Data Processing: PyPDF2, pdf2image, Pandas

- Vector DB: Milvus

- Frontend: Streamlit

- Visualization: Plotly


