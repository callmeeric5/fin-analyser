# Finicial Analyzer
## Description
The Finicial Analyzer is a comprehensive, end-to-end system designed to process and analyze bank statements with advanced OCR, data visualization, and natural language query capabilities. It enables users to extract key information from statements in multiple formats, perform bulk processing, visualize transaction patterns, and interact with data via a conversational interface powered by Retrieval-Augmented Generation (RAG). Supporting multiple large language models (LLMs), including cloud-based APIs and local deployments, the system showcases expertise in data science, machine learning, and natural language processing, making it an ideal portfolio project for data scientist or ML engineer roles.

## Features

- Multi-Format OCR Extraction: Extracts key fields (e.g., amount, date, account number, transaction description) from bank statements in PDF, JPG, or PNG formats using two OCR methods:
  - EasyOCR for rapid, multi-language text extraction.
  - YOLOv8 + Tesseract for precise field detection in complex layouts.


- Batch File Upload: Supports uploading multiple files simultaneously through an intuitive Streamlit web interface, enabling efficient bulk processing.
Automated Data Visualization: Generates interactive visualizations, including:
Line charts for transaction trends over time.
Pie charts for amount distribution by category or range.
Bar charts for transaction frequency by date or type.


- RAG-Powered Natural Language Q&A: Allows users to query extracted data in natural language (e.g., "What is my total spending this month?" or "List all transactions over $500") using LangChain and a vector database (FAISS/Chroma) for context-aware responses.
    - Multi-LLM Support: Integrates multiple LLMs for flexible Q&A:
  - Cloud-based: OpenAI ChatGPT, Google Gemini.
Local: LLaMA, DeepSeek (optimized for GPU/CPU inference).


- Scalable and Modular Design: Organized into modular components (preprocessing, OCR, visualization, RAG, LLM), facilitating maintenance and future enhancements.
