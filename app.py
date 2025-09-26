import os
import io
import re
import json
from typing import List, Dict, Optional, Tuple

import streamlit as st
import requests
from dotenv import load_dotenv
from pypdf import PdfReader


# --- Setup ---
load_dotenv()

APP_TITLE = "Customer Service Chatbot Sandbox"


# --- Utilities ---
def get_openai_client():
    """Return a tuple (client_type, client_obj_or_module) based on installed SDK.

    client_type: "v1" for new SDK (OpenAI class), "legacy" for old SDK, or None if unavailable.
    """
    try:
        # New SDK (>=1.0)
        from openai import OpenAI  # type: ignore

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None, None
        client = OpenAI(api_key=api_key)
        return "v1", client
    except Exception:
        try:
            # Legacy SDK (<1.0)
            import openai  # type: ignore

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return None, None
            openai.api_key = api_key
            return "legacy", openai
        except Exception:
            return None, None


def ensure_session_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # list of {role, content}
    if "system_prompt" not in st.session_state:
        # Initialize with example prompt; users can edit/delete freely
        st.session_state["system_prompt"] = load_example_system_prompt()
    if "model_choice" not in st.session_state:
        st.session_state["model_choice"] = "OpenAI: gpt-3.5-turbo"
    if "raw_request" not in st.session_state:
        st.session_state["raw_request"] = None
    if "kb" not in st.session_state:
        st.session_state["kb"] = None  # dict with {text, chunks, pages, filename}


def load_example_system_prompt() -> str:
    return (
        "You are an empathetic, policy-aware customer service assistant.\n"
        "- Always greet the customer warmly and acknowledge their feelings.\n"
        "- If asking for account/order details, explain why and provide alternatives.\n"
        "- Follow refund and replacement policies; never promise exceptions without guidance.\n"
        "- Offer clear next steps and set expectations for timelines.\n"
        "- Keep responses concise, friendly, and solution-focused.\n"
        "- De-escalate tense situations respectfully and avoid blame.\n"
    )


# --- Prompt formatting for HF instruct model ---
 


# --- Knowledge base helpers ---
WORD_RE = re.compile(r"[A-Za-z0-9']+")
STOPWORDS = set(
    """
    a an and are as at be but by for from has have how i if in into is it its of on or that the their them then there these they this to was we what when where which who why will with you your yours our ours us
    """.split()
)


def extract_pdf_text(file_bytes: bytes) -> Tuple[str, int]:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = len(reader.pages)
        texts = []
        for p in reader.pages:
            try:
                txt = p.extract_text() or ""
            except Exception:
                txt = ""
            texts.append(txt)
        full = "\n\n".join(texts)
        return full, pages
    except Exception:
        return "", 0


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in WORD_RE.findall(text)]


def chunk_text(text: str, max_words: int = 220) -> List[str]:
    # Split by paragraphs first
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: List[str] = []
    for para in paras:
        tokens = _tokenize(para)
        if not tokens:
            continue
        # Slide over long paragraphs
        start = 0
        while start < len(tokens):
            end = min(start + max_words, len(tokens))
            piece_tokens = tokens[start:end]
            # Reconstruct approximate text piece
            piece = " ".join(piece_tokens)
            chunks.append(piece)
            start = end
    # Fallback if nothing
    if not chunks and text.strip():
        tokens = _tokenize(text)
        for i in range(0, len(tokens), max_words):
            chunks.append(" ".join(tokens[i:i + max_words]))
    return chunks


def score_chunk(query_tokens: List[str], chunk_tokens: List[str]) -> float:
    # Simple overlap score with slight length penalty
    if not chunk_tokens:
        return 0.0
    q_set = [t for t in query_tokens if t not in STOPWORDS]
    if not q_set:
        q_set = query_tokens
    counts = {}
    for t in chunk_tokens:
        counts[t] = counts.get(t, 0) + 1
    score = sum(counts.get(t, 0) for t in set(q_set))
    penalty = 1.0 + len(chunk_tokens) / 400.0
    return score / penalty


def retrieve_relevant_chunks(query: str, chunks: List[str], k: int = 3) -> List[str]:
    if not chunks:
        return []
    q_tokens = _tokenize(query)
    scored = []
    for ch in chunks:
        ch_tokens = _tokenize(ch)
        s = score_chunk(q_tokens, ch_tokens)
        if s > 0:
            scored.append((s, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


# --- Model calls ---
def call_openai(messages: List[Dict[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    """Call OpenAI Chat Completions API.

    Returns (assistant_text, error_message). One will be None.
    Also stores raw request payload in session_state for debugging.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "Missing OPENAI_API_KEY. Set it in your environment or .env."

    # Reasonable defaults
    model = "gpt-3.5-turbo"
    temperature = 0.3
    max_tokens = 512

    # Prepare raw payload for display
    raw_payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    st.session_state["raw_request"] = {"provider": "openai", "payload": raw_payload}

    client_type, client = get_openai_client()
    if client_type is None:
        return None, "OpenAI SDK not available. Ensure 'openai' is installed."

    try:
        if client_type == "v1":
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content if resp and resp.choices else None
        else:  # legacy
            resp = client.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp["choices"][0]["message"]["content"] if resp and resp.get("choices") else None

        if not text or not text.strip():
            return None, "Empty response from OpenAI."
        return text.strip(), None
    except Exception as e:
        return None, f"OpenAI error: {e}"


 


def call_groq(messages: List[Dict[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    """Call Groq Chat Completions API for gemma2-9b-it.

    Returns (assistant_text, error_message). Also stores raw request in session state.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None, "Missing GROQ_API_KEY. Set it in your environment or .env."

    model = "gemma2-9b-it"
    temperature = 0.3
    max_tokens = 512

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    st.session_state["raw_request"] = {"provider": "groq", "payload": payload}

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            return None, f"Groq API error {resp.status_code}: {resp.text}"
        data = resp.json()
        text = None
        if isinstance(data, dict) and data.get("choices"):
            choice = data["choices"][0]
            # OpenAI-compatible shape
            msg = choice.get("message") if isinstance(choice, dict) else None
            if msg and isinstance(msg.get("content"), str):
                text = msg["content"]
        if not text or not text.strip():
            return None, "Empty response from Groq."
        return text.strip(), None
    except Exception as e:
        return None, f"Groq error: {e}"


# --- UI ---
def sidebar():
    with st.sidebar:
        st.header("Settings")

        # Model selector
        model = st.selectbox(
            "Model",
            [
                "OpenAI: gpt-3.5-turbo",
                "Groq: gemma2-9b-it",
            ],
            index=0 if st.session_state["model_choice"].startswith("OpenAI") else 1,
        )
        st.session_state["model_choice"] = model

        # System prompt editor
        st.subheader("System Prompt")
        st.session_state["system_prompt"] = st.text_area(
            "Edit the system prompt",
            value=st.session_state.get("system_prompt", ""),
            height=180,
        )

        # Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset Conversation", use_container_width=True):
                st.session_state["messages"] = []
        with col2:
            if st.button("Load Example Prompt", use_container_width=True):
                st.session_state["system_prompt"] = load_example_system_prompt()

        # API key status
        st.subheader("API Keys")
        openai_ok = bool(os.getenv("OPENAI_API_KEY"))
        groq_ok = bool(os.getenv("GROQ_API_KEY"))
        st.caption(f"OpenAI: {'✅ set' if openai_ok else '⚠️ missing'}")
        st.caption(f"Groq: {'✅ set' if groq_ok else '⚠️ missing'}")

        # (Former Test Inputs removed)

        # Knowledge base uploader
        with st.expander("Knowledge Base (PDF)", expanded=False):
            kb = st.session_state.get("kb")
            uploaded = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)
            c1, c2 = st.columns([1, 1])
            with c1:
                clear_kb = st.button("Clear KB", use_container_width=True)
            with c2:
                process_now = st.button("Process PDF", use_container_width=True)

            if clear_kb:
                st.session_state["kb"] = None

            if uploaded and (process_now or kb is None):
                with st.spinner("Extracting text from PDF…"):
                    content, pages = extract_pdf_text(uploaded.read())
                if not content.strip():
                    st.warning("Could not extract text from the PDF.")
                else:
                    chunks = chunk_text(content)
                    st.session_state["kb"] = {
                        "filename": uploaded.name,
                        "pages": pages,
                        "text": content,
                        "chunks": chunks,
                        "n_chunks": len(chunks),
                    }

            kb = st.session_state.get("kb")
            if kb:
                st.success(f"Loaded {kb['filename']} — {kb['pages']} pages, {kb['n_chunks']} chunks")
                st.caption("The most relevant excerpts will be provided to the model with your question.")
            else:
                st.info("No knowledge base loaded.")

        # Debug tools
        with st.expander("Debug", expanded=False):
            if st.button("Test Groq Connectivity"):
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    st.error("GROQ_API_KEY is missing. Add it to your environment or Secrets.")
                else:
                    url = "https://api.groq.com/openai/v1/chat/completions"
                    payload = {
                        "model": "gemma2-9b-it",
                        "messages": [
                            {"role": "system", "content": "Connectivity test"},
                            {"role": "user", "content": "Say 'hello'"},
                        ],
                        "max_tokens": 4,
                        "temperature": 0.1,
                    }
                    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}
                    try:
                        r = requests.post(url, headers=headers, json=payload, timeout=60)
                        if r.status_code != 200:
                            st.error(f"Groq API error {r.status_code}: {r.text[:500]}")
                        else:
                            st.success("Groq connectivity OK")
                            try:
                                st.json(r.json())
                            except Exception:
                                st.write(r.text)
                    except Exception as e:
                        st.error(f"Groq connectivity test failed: {e}")


def render_chat():
    # Render prior messages
    for msg in st.session_state["messages"]:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        if hasattr(st, "chat_message"):
            with st.chat_message(role):
                st.markdown(content)
        else:
            # Fallback render
            st.markdown(f"**{role.title()}:** {content}")


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="💬")
    ensure_session_state()

    st.title(APP_TITLE)
    sidebar()

    render_chat()

    # Chat input
    user_input = st.chat_input("Type your message…") if hasattr(st, "chat_input") else st.text_input("Your message")

    # Handle sending
    if user_input:
        # Append user message to history
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # Prepare request based on selected model
        model_choice = st.session_state["model_choice"]
        base_system_prompt = st.session_state["system_prompt"] or ""

        # RAG: build contextual system prompt using KB
        effective_system_prompt = base_system_prompt
        kb = st.session_state.get("kb")
        if kb and kb.get("chunks"):
            top_chunks = retrieve_relevant_chunks(user_input, kb["chunks"], k=3)
            if top_chunks:
                context = "\n\n".join(f"- {c}" for c in top_chunks)
                effective_system_prompt = (
                    f"{base_system_prompt}\n\n"
                    "Use the following knowledge base excerpts if relevant. "
                    "If the KB does not contain the answer, say so or ask for clarification.\n"
                    "[Knowledge Base Excerpts]\n"
                    f"{context}"
                )

        if model_choice.startswith("OpenAI"):
            # Prepend system message
            api_messages = [{"role": "system", "content": effective_system_prompt}] + st.session_state["messages"]

            with st.spinner("Calling OpenAI…"):
                assistant_text, error = call_openai(api_messages)

        else:
            # Groq uses OpenAI-compatible chat API
            api_messages = [{"role": "system", "content": effective_system_prompt}] + st.session_state["messages"]
            with st.spinner("Calling Groq…"):
                assistant_text, error = call_groq(api_messages)

        if error:
            st.error(error)
        else:
            st.session_state["messages"].append({"role": "assistant", "content": assistant_text})

        # Re-render updated conversation
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

    # Raw request viewer
    with st.expander("Show raw request"):
        raw = st.session_state.get("raw_request")
        if not raw:
            st.info("No request sent yet.")
        else:
            provider = raw.get("provider")
            if provider == "openai":
                st.caption("OpenAI Chat Completions payload")
                st.json(raw.get("payload"))
            elif provider == "groq":
                st.caption("Groq Chat Completions payload")
                st.json(raw.get("payload"))
            else:
                st.json(raw)


if __name__ == "__main__":
    main()
