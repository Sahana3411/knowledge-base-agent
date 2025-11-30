
# Knowledge Base Agent — AI-Powered Document Question-Answering System

## Overview

This project is an AI-driven knowledge assistant that allows users to upload documents and ask questions directly from the content. The system extracts text from PDFs, DOCX, TXT and images (OCR), and then uses LLMs (OpenAI, Google Gemini or Groq) to answer user queries with context awareness.
The agent supports real-time chat history, one-click responses, and works entirely with user-provided API keys (session-only), ensuring privacy and safety.

The purpose of this agent is to serve as a lightweight, user-friendly knowledge retrieval system suitable for academic use, research analysis, and AI-assisted content review.

---

## Features

### Document Understanding

* Upload multiple files: PDF, DOCX, TXT, PNG, JPG
* Automatic text extraction (PDF/DOCX/TXT parsing + OCR for images)
* Combined document corpus used for answering queries

### AI-Powered Chat

* Supports OpenAI, Google Gemini and Groq
* No API key stored in the project; user pastes key in UI (session-only)
* Maintains multi-turn chat history similar to ChatGPT
* Answers generated strictly from the documents uploaded
* Clear history & export chat options

### User Interface

* Built with Streamlit (dark-blue theme)
* Single-click Send
* Left-aligned user messages and right-aligned assistant messages
* File management (delete uploaded files)
* Clean layout without unnecessary preview clutter

### Safety & Privacy

* Keys are never logged or saved
* All processing happens locally except the LLM API call
* No backend database; session-based storage

---

## Tech Stack

### Core Framework

* Streamlit – UI, session management, and rendering

### AI / LLM Providers

* OpenAI (Chat/Responses API)
* Google Gemini (REST API)
* Groq (OpenAI-compatible endpoint)

### Document Processing

* PyPDF2 – PDF extraction
* python-docx – DOCX extraction
* pytesseract + Pillow – OCR for images
* Plain text parsing

### Programming Language

* Python 3.10+ compatible
* Minimal dependencies for fast installation

---

## APIs Used

* OpenAI Chat Completions / Responses API
* Google Generative Language API (Gemini)
* Groq’s OpenAI-compatible Chat Completions endpoint

All keys must be manually provided by the user in the sidebar. No key is bundled or stored.

---

## Setup & Run Instructions

### 1. Clone the repository

```
git clone <your-repo-url>
cd knowledge-base-agent
```

### 2. Create a virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. (Optional) Install Tesseract OCR for image-to-text

Download Windows installer:
[https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

### 5. Run the Streamlit app

```
streamlit run app.py
```


### 6. Architecture Diagram

<p align="center">
<svg width="860" height="520" viewBox="0 0 860 520" xmlns="http://www.w3.org/2000/svg">

  <!-- BACKGROUND -->
  <rect width="860" height="520" fill="#0B0F19" rx="18"/>

  <!-- TITLE -->
  <text x="430" y="40" fill="#FFFFFF" font-size="26" font-family="Arial" text-anchor="middle">
    Knowledge Agent – System Architecture
  </text>

  <!-- UPLOAD BLOCK -->
  <rect x="40" y="90" width="220" height="120" fill="#1C2333" stroke="#4A90E2" stroke-width="2" rx="12"/>
  <text x="150" y="130" fill="#FFFFFF" font-size="18" font-family="Arial" text-anchor="middle">Document Upload</text>
  <text x="150" y="160" fill="#B8C7E0" font-size="14" font-family="Arial" text-anchor="middle">
    PDF / DOCX / TXT / Images
  </text>

  <!-- EXTRACTORS -->
  <rect x="330" y="90" width="220" height="120" fill="#1C2333" stroke="#4A90E2" stroke-width="2" rx="12"/>
  <text x="440" y="130" fill="#FFFFFF" font-size="18" font-family="Arial" text-anchor="middle">
    Text Extractors
  </text>
  <text x="440" y="160" fill="#B8C7E0" font-size="14" font-family="Arial" text-anchor="middle">
    extract_pdf • extract_docx • extract_txt • OCR
  </text>

  <!-- CORPUS -->
  <rect x="620" y="90" width="220" height="120" fill="#1C2333" stroke="#4A90E2" stroke-width="2" rx="12"/>
  <text x="730" y="130" fill="#FFFFFF" font-size="18" font-family="Arial" text-anchor="middle">
    Corpus Builder
  </text>
  <text x="730" y="160" fill="#B8C7E0" font-size="14" font-family="Arial" text-anchor="middle">
    Combined Text (40k chars)
  </text>

  <!-- LLM ROUTER -->
  <rect x="330" y="260" width="220" height="120" fill="#1C2333" stroke="#4A90E2" stroke-width="2" rx="12"/>
  <text x="440" y="300" fill="#FFFFFF" font-size="18" font-family="Arial" text-anchor="middle">
    LLM Router
  </text>
  <text x="440" y="330" fill="#B8C7E0" font-size="14" font-family="Arial" text-anchor="middle">
    OpenAI • Gemini • Groq
  </text>

  <!-- CHAT UI -->
  <rect x="40" y="260" width="220" height="120" fill="#1C2333" stroke="#4A90E2" stroke-width="2" rx="12"/>
  <text x="150" y="300" fill="#FFFFFF" font-size="18" font-family="Arial" text-anchor="middle">
    Chat UI (Streamlit)
  </text>
  <text x="150" y="330" fill="#B8C7E0" font-size="14" font-family="Arial" text-anchor="middle">
    Chat history • Chat bubbles
  </text>

  <!-- STORAGE -->
  <rect x="620" y="260" width="220" height="120" fill="#1C2333" stroke="#4A90E2" stroke-width="2" rx="12"/>
  <text x="730" y="300" fill="#FFFFFF" font-size="18" font-family="Arial" text-anchor="middle">
    Session Storage
  </text>
  <text x="730" y="330" fill="#B8C7E0" font-size="14" font-family="Arial" text-anchor="middle">
    Chat History (Session-only)
  </text>

  <!-- CONNECTION ARROWS -->
  <line x1="260" y1="150" x2="330" y2="150" stroke="#4A90E2" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="550" y1="150" x2="620" y2="150" stroke="#4A90E2" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="150" y1="210" x2="150" y2="260" stroke="#4A90E2" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="440" y1="210" x2="440" y2="260" stroke="#4A90E2" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="730" y1="210" x2="730" y2="260" stroke="#4A90E2" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="260" y1="320" x2="330" y2="320" stroke="#4A90E2" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="550" y1="320" x2="620" y2="320" stroke="#4A90E2" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- ARROW MARKER -->
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="5" refY="3" orient="auto">
      <path d="M0,0 L0,6 L9,3 z" fill="#4A90E2" />
    </marker>
  </defs>

</svg>
</p>


### 7. Using the Application

In the sidebar:

1. Upload documents
2. Select LLM provider (OpenAI / Gemini / Groq)
3. Paste your API key (session-only, never stored)
4. Type your question and press Send
5. Chat history will grow automatically
6. Export full chat from sidebar if required

---

## Limitations

* Requires a valid and active API key from your chosen provider
* Offline answers are only keyword-based and may be approximate
* Depends on proper document text extraction (OCR may vary in accuracy)
* Gemini may return 404/400 errors if the user’s API key lacks model access
* Groq model names must follow current availability (old models deprecated)

## Potential Improvements (Short & Professional)

* Vector Search Integration: Add FAISS / ChromaDB to improve accuracy by retrieving semantically relevant document chunks.
* Multi-Model Auto-Fallback: Automatically switch to a backup provider (OpenAI → Groq → Gemini) during outages.
* UI Enhancements: Add animations, typing indicators, message reactions, and custom themes.
* Advanced Document Processing: Support tables, scanned PDFs (OCR), and multi-language extraction.
* User Authentication: Optional login system to allow personalized histories and project separation.
* Conversation Memory Optimization: Implement message summarization to keep long chats efficient.
* File Preview Panel: Inline previews for PDFs, DOCX and images inside chat window.
