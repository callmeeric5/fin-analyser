from gradio_client import Client, handle_file
import json
import re


def call_qwen_api(path: str):
    client = Client("callmeeric5/Qwen3B-Invoice-Receipt")
    result = client.predict(
        image=handle_file(path),
        custom_instruction=None,
        api_name="/predict",
    )
    return result


def call_qdrant_api(text: str, model: str):
    """
    Calls the Qdrant API for other LLM models.
    """
    # Placeholder for Qdrant API call logic
    # You will need to replace this with your actual Qdrant client implementation
    print(f"Calling Qdrant API for model: {model} with text: {text[:100]}...")
    return {"result": "Structured data from Qdrant API"}
