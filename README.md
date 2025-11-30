
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

    
                        ┌────────────────────────────┐
                        │        User Interface      │
                        │(Streamlit ChatUI + Sidebar)│
                        └─────────────┬──────────────┘
                                      │
         ┌────────────────────────────┼─────────────────────────────┐
         │                            │                             │
┌────────▼────────┐        ┌──────────▼───────────┐        ┌─────────▼─────────┐
│ File Upload UI  │        │  Chat History Layer  │        │   Settings Panel  │
│ (PDF/DOCX/TXT)  │        │ (Session State)      │        │ Provider + API Key│
└────────┬────────┘        └──────────┬───────────┘        └─────────┬─────────┘
         │                            │                              │
         ▼                            ▼                              ▼
 ┌────────────────────┐       ┌──────────────────────┐        ┌────────────────────┐
 │ Document Extractors│       │ Prompt Builder       │        │ Model Selector     │
 │ (PyPDF2, DOCX, OCR)│       │ Corpus + History     │        │ OpenAI/Gemini/Groq │
 └─────────┬──────────┘       └──────────┬───────────┘        └─────────┬──────────┘
           │                             │                              │
           ▼                             ▼                              ▼
 ┌──────────────────────┐      ┌─────────────────────────┐      ┌──────────────────────┐
 │   Offline Engine     │      │   LLM Query Engine      │      │   API Integrations   │
 │ (keyword snippet)    │      │  (Chat / Responses)     │      │   OpenAI / Gemini /  │
 └─────────┬────────────┘      └──────────┬──────────────┘      │        Groq          │
           │                              │                     └─────────┬────────────┘
           ▼                              ▼                               ▼
   ┌────────────────────────────────────────────────────────────────────────────────┐
   │                              Assistant Response                                │
   └────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                           Displayed back in UI as chat


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
