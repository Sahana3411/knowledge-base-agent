import os
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import pytesseract

def extract_pdf(path):
    try:
        reader = PdfReader(path)
        texts = [p.extract_text() or '' for p in reader.pages]
        return '\n'.join(texts)
    except Exception as e:
        return ''

def extract_docx(path):
    try:
        doc = Document(path)
        return '\n'.join([p.text for p in doc.paragraphs])
    except Exception:
        return ''

def extract_txt(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        return ''

def extract_image(path):
    try:
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    except Exception:
        return ''
