# app.py
# On-demand LLM Knowledge Agent — single-send, dark-blue UI, chat history, delete files
# Paste API keys in sidebar (session-only). No keys are saved to disk.
import os
import time
import json
import requests
import streamlit as st
from pathlib import Path
from extractors import extract_pdf, extract_docx, extract_txt, extract_image

# -----------------------
# Page config + dark blue CSS
# -----------------------
st.set_page_config(page_title="Knowledge Based Agent ", layout="wide")
st.markdown("""
<style>

:root {
  --bg: #250D44;
  --card: #171C21;
  --accent: #1e90ff;
  --text: #e6f0ff;
  --muted: #bcd3f5;
}

/* Input boxes */
.stTextInput input {
    background: #ffffff !important;
    color: #000000 !important;
    caret-color: #1e90ff !important;
    border-radius: 6px;
}

textarea {
    background: #ffffff !important;
    color: #000000 !important;
    caret-color: #1e90ff !important;
    border-radius: 6px;
}

/* Background */
html, body, .main, .stApp {
    background: linear-gradient(180deg,var(--bg), #07203a);
    color: var(--text);
}

/* Buttons */
.stButton>button {
    background-color: var(--accent);
    color: white;
    border-radius: 8px;
}

/* Sidebar */
.stSidebar {
    background-color: var(--card);
    color: var(--text);
    padding: 12px;
    border-radius: 8px;
}

/* Upload / misc styling */
.stDownloadButton>button {
    background-color: #0b78d1;
    color: white;
}

.stFileUploader label {
    color: var(--text);
}

/* ---------- CHAT UI ONLY ---------- */
.chat-container {
    max-height: 420px;
    overflow-y: auto;
    padding: 12px;
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.15);
    margin-bottom: 15px;
}

/* User bubble */
.chat-bubble-user {
    background: #e6f0ff;
    color: #000;
    padding: 10px 14px;
    border-radius: 14px;
    max-width: 60%;
    margin-bottom: 8px;
    border: 1px solid #c9dbf8;
    text-align: left;
}

/* Assistant bubble */
.chat-bubble-assistant {
    background: #d7e6ff;
    color: #000;
    padding: 10px 14px;
    border-radius: 14px;
    max-width: 60%;
    margin-bottom: 8px;
    margin-left: auto;
    border: 1px solid #b8cef0;
    text-align: left;
}

/* Timestamps */
.timestamp {
    font-size: 12px;
    color: #bcd3f5;
    margin-top: -3px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


st.title("Knowledge Base Agent — Chat")
st.write("Upload documents, paste an API key in the sidebar (session-only), and chat. History retained in session.")
st.markdown("---")

# -----------------------
# Folders
# -----------------------
UPLOAD_DIR = "docs"
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# -----------------------
# Sidebar: uploads + provider + key + model override
# -----------------------
st.sidebar.header("Documents & LLM Settings")

# Upload files (saved to docs/)
uploaded = st.sidebar.file_uploader(
    "Drag & drop (PDF, DOCX, TXT, PNG, JPG) — multiple OK",
    accept_multiple_files=True
)
if uploaded:
    saved = []
    for f in uploaded:
        out = Path(UPLOAD_DIR) / f.name
        base, ext = out.stem, out.suffix
        i = 1
        while out.exists():
            out = Path(UPLOAD_DIR) / f"{base}_{i}{ext}"
            i += 1
        with open(out, "wb") as wf:
            wf.write(f.read())
        saved.append(out.name)
    st.sidebar.success(f"Saved {len(saved)} file(s).")

st.sidebar.markdown("---")

provider = st.sidebar.selectbox("LLM Provider (paste key below)", ["OpenAI", "Google Gemini", "Groq"])
api_key = st.sidebar.text_input("Paste API key (session-only)", type="password")

# model overrides (helps if default model not available)
openai_model = st.sidebar.text_input("OpenAI model (optional)", value="gpt-4o-mini") if provider == "OpenAI" else None
gemini_model = st.sidebar.text_input("Gemini model (optional)", value="gemini-1.5-flash") if provider == "Google Gemini" else None
groq_url = st.sidebar.text_input("Groq full endpoint (optional)", placeholder="https://api.groq.com/openai/v1/chat/completions") if provider == "Groq" else None
groq_model = st.sidebar.text_input("Groq model (optional)", value="llama-3.1-8b-instant") if provider == "Groq" else None

st.sidebar.markdown("---")
st.sidebar.header("Manage uploaded files")
docs = sorted(os.listdir(UPLOAD_DIR))
if docs:
    for d in docs:
        file_path = os.path.join(UPLOAD_DIR, d)
        c1, c2 = st.sidebar.columns([4,1])
        c1.write(d)
        if c2.button("🗑", key=f"del_{d}"):
            try:
                os.remove(file_path)
                st.sidebar.success(f"Deleted {d}")
                # try rerun; if experimental_rerun not available, instruct user to refresh
                try:
                    st.rerun()
                except Exception:
                    st.sidebar.info("File deleted — refresh the page to update the list.")
            except Exception as e:
                st.sidebar.error(f"Could not delete {d}: {e}")
else:
    st.sidebar.write("No documents uploaded.")

st.sidebar.markdown("---")
st.sidebar.caption("Paste keys only for the session. Keys are not written to disk by this app.")

# -----------------------
# Session: chat history
# -----------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role":"user"/"assistant","content":..., "ts":...}

# Sidebar chat controls
if st.sidebar.button("Clear history"):
    st.session_state.history = []
    try:
        st.rerun()
    except Exception:
        st.sidebar.success("History cleared (refresh page if it still appears).")

if st.sidebar.button("Export history (JSON)"):
    data = json.dumps(st.session_state.history, indent=2, ensure_ascii=False)
    st.sidebar.download_button("Download JSON", data, file_name="chat_history.json", mime="application/json")

st.sidebar.markdown("---")

# -----------------------
# Build corpus (no preview)
# -----------------------
def build_corpus(limit_chars=80000):
    parts = []
    for fname in sorted(os.listdir(UPLOAD_DIR)):
        p = os.path.join(UPLOAD_DIR, fname)
        try:
            if fname.lower().endswith(".pdf"):
                parts.append(extract_pdf(p))
            elif fname.lower().endswith(".docx"):
                parts.append(extract_docx(p))
            elif fname.lower().endswith(".txt"):
                parts.append(extract_txt(p))
            elif fname.lower().endswith((".png", ".jpg", ".jpeg")):
                parts.append(extract_image(p))
        except Exception:
            pass
    full = "\n\n".join([t for t in parts if t]).strip()
    return full[:limit_chars]

corpus = build_corpus()

# -----------------------
# Offline snippet search (helpful if no API key)
# -----------------------
def offline_answer(corpus_text, question_text, top_k=3):
    if not corpus_text.strip():
        return None, []
    q_words = [w.lower() for w in question_text.split() if len(w) > 2]
    sentences = [s.strip() for s in corpus_text.replace("\n", " ").split(".") if s.strip()]
    scored = []
    for sent in sentences:
        s_lower = sent.lower()
        score = sum(1 for w in q_words if w in s_lower)
        if score > 0:
            scored.append((score, sent))
    scored.sort(key=lambda x: x[0], reverse=True)
    snippets = [s for _, s in scored[:top_k]]
    answer = " ".join(snippets) if snippets else None
    return answer, snippets

# -----------------------
# Provider callers (robust)
# -----------------------
def call_openai(api_key, messages_or_prompt, model=None, max_tokens=800, timeout=60):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # Try Responses API first (single input string)
    try:
        url = "https://api.openai.com/v1/responses"
        payload = {"model": model or "gpt-4o-mini", "input": messages_or_prompt if isinstance(messages_or_prompt, str) else "".join([m.get("content","") for m in messages_or_prompt]), "max_output_tokens": max_tokens}
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        if "output_text" in j:
            return j["output_text"]
        out = []
        for block in j.get("output", []):
            for c in block.get("content", []):
                if "text" in c:
                    out.append(c["text"])
        if out:
            return "\n".join(out)
    except requests.HTTPError:
        raise
    except Exception:
        pass

    # Fallback to chat completions
    if isinstance(messages_or_prompt, list):
        url = "https://api.openai.com/v1/chat/completions"
        payload = {"model": model or "gpt-4o-mini", "messages": messages_or_prompt, "temperature": 0.0, "max_tokens": max_tokens}
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]
    else:
        url = "https://api.openai.com/v1/chat/completions"
        payload = {"model": model or "gpt-4o-mini", "messages": [{"role":"user","content": messages_or_prompt}], "temperature":0.0, "max_tokens": max_tokens}
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]

def call_gemini(api_key, prompt_text, model_name="gemini-1.5-flash", max_tokens=800, timeout=60):
    base = "https://generativelanguage.googleapis.com/v1beta/models"
    url = f"{base}/{model_name}:generate?key={api_key}"
    payload = {"prompt": {"text": prompt_text}, "maxOutputTokens": max_tokens}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if "candidates" in j and j["candidates"]:
        cand = j["candidates"][0]
        content = cand.get("content", {})
        parts = content.get("parts", [])
        if parts:
            return parts[0].get("text", "")
    if "output" in j and j["output"]:
        return j["output"][0].get("content", [{}])[0].get("text", "")
    return str(j)

def call_groq(api_key, prompt_text, api_url=None, model="llama-3.1-8b-instant", max_tokens=800, timeout=60):
    """
    Groq caller using OpenAI-compatible chat completions endpoint.
    If user provides a full endpoint in sidebar (groq_url), that is used.
    """
    url = api_url if api_url else "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.0,
        "max_tokens": max_tokens
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    # common OpenAI-compatible response
    if isinstance(j, dict) and "choices" in j and j["choices"]:
        return j["choices"][0]["message"].get("content") or j["choices"][0]["message"].get("content", "")
    # fallback: stringify
    return str(j)

# -----------------------
# Prompt building
# -----------------------
def build_system_instruction():
    return (
        "You are a helpful knowledge assistant. Use ONLY the provided document text to answer user questions. "
        "If the answer is not present, reply: 'Not found in provided documents.' Keep answers little concise."
    )

def build_prompt_from_history(corpus_text, history, user_message):
    sys_instr = build_system_instruction()
    recent = history[-10:] if history else []
    hist_text = ""
    for m in recent:
        role = m.get("role", "user")
        content = m.get("content", "")
        hist_text += f"\n{role.upper()}:\n{content}\n"
    prompt = f"{sys_instr}\n\nDOCUMENTS:\n{corpus_text}\n\nCHAT HISTORY:{hist_text}\nUSER:\n{user_message}\n\nASSISTANT:"
    return prompt

def history_to_openai_messages(history):
    system = {"role":"system", "content": build_system_instruction()}
    msgs = [system]
    for m in history:
        msgs.append({"role": m["role"], "content": m["content"]})
    return msgs

# -----------------------
# Main chat UI (single send)
# -----------------------

for msg in st.session_state.history:
    role = msg.get("role")
    content = msg.get("content")
    ts = msg.get("ts", time.time())
    timestr = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime(ts))

    if role == "user":
        st.markdown(f"""
        <div class="chat-bubble-user"><b>You</b><br>{content}</div>
        <div class="timestamp">{timestr}</div>
        """, unsafe_allow_html=True)

    elif role == "assistant":
        st.markdown(f"""
        <div class="chat-bubble-assistant"><b>Assistant</b><br>{content}</div>
        <div class="timestamp" style="text-align:right;">{timestr}</div>
        """, unsafe_allow_html=True)


# Input + Send
user_msg = st.text_input("Type your message and click Send", key="user_input")

# Create a row: Send + Export buttons together in one line
send_col, export_col = st.columns([4, 1])

with send_col:
    send_clicked = st.button("Send")


# ---------- SEND BUTTON ----------
if send_clicked:
    if not user_msg.strip():
        st.warning("Please type something before sending.")
    else:
        # append user message to history
        st.session_state.history.append({"role":"user", "content": user_msg, "ts": time.time()})
        data = json.dumps(st.session_state.history, indent=2, ensure_ascii=True)

        # Try offline snippet first if no API key
        offline_ans, snippets = offline_answer(corpus, user_msg, top_k=5)

        if offline_ans and not api_key:
            st.session_state.history.append({"role":"assistant", "content": offline_ans, "ts": time.time()})
            data = json.dumps(st.session_state.history, indent=2, ensure_ascii=False)
            try:
                st.rerun()
            except Exception:
                st.success("Offline answer added. Refresh to see updated chat.")

        else:
            if not api_key:
                st.error("No API key provided in sidebar. Paste an API key to call an LLM or rely on offline snippets.")
            else:
                st.info("Contacting LLM — please wait...")
                prompt_payload = build_prompt_from_history(corpus, st.session_state.history, user_msg)

                try:
                    if provider == "OpenAI":
                        messages = history_to_openai_messages(st.session_state.history)
                        ans = call_openai(api_key, messages if messages else prompt_payload, model=openai_model)
                    elif provider == "Google Gemini":
                        model_name = gemini_model or "gemini-1.5-flash"
                        ans = call_gemini(api_key, prompt_payload, model_name=model_name)
                    else:  # Groq
                        ans = call_groq(api_key, prompt_payload, api_url=groq_url if groq_url else None, model=groq_model)

                    # store assistant reply
                    st.session_state.history.append({"role":"assistant", "content": ans, "ts": time.time()})
                    try:
                        st.rerun()
                    except Exception: 
                        st.success()
                          
                except requests.HTTPError as he:
                    code = he.response.status_code
                    body = ""
                    try:
                        body = he.response.json()
                    except Exception:
                        body = he.response.text

                    if code == 429:
                        st.error("Rate limit (429). Check billing or try later. Response: " + str(body))
                    elif code == 404:
                        st.error("404 Not Found — model endpoint invalid or your key lacks access. Details: " + str(body))
                    elif code == 400:
                        st.error("400 Bad Request: " + str(body))
                    else:
                        st.error(f"HTTP error {code}: {body}")

                except requests.exceptions.SSLError as sse:
                    st.error(f"SSL error: {sse}. Check network/proxy/TLS.")

                except requests.exceptions.ConnectionError as ce:
                    st.error(f"Connection error: {ce}. Check internet/DNS or provider endpoint.")

                except Exception as e:
                    st.error(f"LLM call failed: {e}")

st.markdown("---")
st.caption("Chat history lives only in session. Paste keys in sidebar (session-only). Do NOT commit keys.")
