from PyPDF2 import PdfReader
import os
import config

os.environ["OPENAI_API_KEY"] = config.api_key

reader = PdfReader("Demian.pdf")

raw_text = ""

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text


