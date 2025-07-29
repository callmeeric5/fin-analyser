from gradio_client import Client, handle_file
import json
import re
from groq import Groq
from dotenv import load_dotenv
import os
import base64
from google import genai
from google.genai import types

load_dotenv()
groq_key = os.environ.get("GROQ_API_KEY")
qwen_key = os.environ.get("GROQ_API_KEY")
gemini_key = os.environ.get("GEMINI_API_KEY")
schema_dict = {
    "document_type": "",
    "store_name": "",
    "store_address": "",
    "store_phone": "",
    "date": "",
    "time": "",
    "invoice_no": "",
    "invoice_date": "",
    "seller": "",
    "client": "",
    "seller_tax_id": "",
    "client_tax_id": "",
    "iban": "",
    "items": [
        {
            "item_name": "",
            "item_desc": "",
            "item_key": "",
            "item_quantity": "",
            "item_net_price": "",
            "item_value": "",
            "item_net_worth": "",
            "item_vat": "",
            "item_gross_worth": "",
        }
    ],
    "subtotal": "",
    "tax": "",
    "tips": "",
    "total": "",
    "total_net_worth": "",
    "total_vat": "",
    "total_gross_worth": "",
}
sys_prompt = f"""
You are specialized in extracting information from invoices and receipts.
Your task is to extract the information from the provided document image in the following valid JSON format.
Always use this exact structure. The 'items' field MUST be a list of objects, each with these keys: 'item_name', 'item_desc', 'item_key', 'item_quantity', 'item_net_price', 'item_value', 'item_net_worth', 'item_vat', 'item_gross_worth'. 
If any key is missing, leave it as an empty string.
Do NOT add any comments, explanation, or preamble—output exactly one valid JSON object as below:
{json.dumps(schema_dict, indent=2)}
Fill only the keys that are available.
Output ONLY a single valid JSON object. Do not include any explanation, role, or any extra text.
"""


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def call_qwen_api(path: str):
    client = Client("callmeeric5/Qwen3B-Invoice-Receipt")
    result = client.predict(
        image=handle_file(path),
        custom_instruction=None,
        api_name="/predict",
    )
    return result

def call_groq_api(path: str):
    base64_image = encode_image(path)
    client = Groq(api_key=groq_key)
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=[
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    }
                ],
            },
        ],
        temperature=1,
        max_completion_tokens=1024,
    )
    data = completion.choices[0].message.content
    print(data)
    return json.loads(data)


def call_gemini_api(path: str):
    base64_image = encode_image(path)
    client = genai.Client(
        api_key=gemini_key,
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(
                data=base64_image,
                mime_type="image/jpeg",
            ),
            sys_prompt,
        ],
        config={"response_mime_type": "application/json"},
    )
    data = response.text
    return json.loads(data)
