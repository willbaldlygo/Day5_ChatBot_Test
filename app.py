import os
import io
import re
import json
from typing import List, Dict, Optional, Tuple

import streamlit as st
import requests
from dotenv import load_dotenv
from pypdf import PdfReader
import time


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
def build_mistral_instruct_prompt(system_prompt: str, history: List[Dict[str, str]], new_user_message: str) -> str:
    """Format conversation into a single prompt for Mistral-Instruct models.

    History is a list of dicts [{role, content}] with roles in {"user", "assistant"}.
    """
    def sanitize(text: str) -> str:
        return text.replace("</s>", "").strip()

    sys_block = f"<<SYS>>\n{sanitize(system_prompt)}\n<</SYS>>\n\n"

    prompt_parts: List[str] = []
    prompt_parts.append(f"<s>[INST] {sys_block}")

    # Iterate over history as pairs (user -> assistant)
    i = 0
    while i < len(history):
        u = history[i]["content"] if history[i]["role"] == "user" else ""
        a = ""
        if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
            a = history[i + 1]["content"]
            i += 2
        else:
            i += 1
        u = sanitize(u)
        a = sanitize(a)
        if u:
            prompt_parts.append(f"{u} [/INST]\n{a} </s>\n<s>[INST] ")

    # Final incoming user message
    prompt_parts.append(f"{sanitize(new_user_message)} [/INST]")

    return "".join(prompt_parts)


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


def call_huggingface(system_prompt: str, history: List[Dict[str, str]], new_user_message: str) -> Tuple[Optional[str], Optional[str]]:
    """Call Hugging Face Inference API for Mistral-7B-Instruct-v0.2.

    Returns (assistant_text, error_message).
    Stores raw request (prompt + params) for display.
    """
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        return None, "Missing HF_API_KEY. Set it in your environment or .env."

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    url = f"https://api-inference.huggingface.co/models/{model_id}"

    prompt = build_mistral_instruct_prompt(system_prompt, history, new_user_message)
    params = {
        "max_new_tokens": 256,
        "temperature": 0.3,
        "top_p": 0.95,
        "do_sample": True,
        "return_full_text": False,
    }
    payload = {"inputs": prompt, "parameters": params, "options": {"wait_for_model": True}}

    # Store raw request snapshot
    st.session_state["raw_request"] = {
        "provider": "huggingface",
        "model": model_id,
        "prompt": prompt,
        "parameters": params,
    }

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Poll on 503/model loading up to ~180s
        deadline = time.time() + 180
        attempt = 0
        while True:
            attempt += 1
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            # Handle non-OK statuses
            if resp.status_code == 503:
                # Model loading / warming up
                try:
                    info = resp.json()
                except Exception:
                    info = {"error": resp.text}
                est = float(info.get("estimated_time", 5.0)) if isinstance(info, dict) else 5.0
                if time.time() + est > deadline:
                    return None, f"HF API model still loading after multiple attempts: {info}"
                # brief wait then retry
                time.sleep(min(max(est, 2.0), 10.0))
                continue
            if resp.status_code != 200:
                return None, f"HF API error {resp.status_code}: {resp.text}"

            # Parse successful response
            try:
                data = resp.json()
            except Exception as e:
                return None, f"HF API invalid JSON: {e}"

            # Supported shapes:
            # - list[{generated_text: str}]
            # - {generated_text: str}
            # - plain string
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict) and "generated_text" in first:
                    gen = first.get("generated_text")
                    if isinstance(gen, str) and gen.strip():
                        return gen.strip(), None
            if isinstance(data, dict):
                if "generated_text" in data and isinstance(data["generated_text"], str):
                    gen = data["generated_text"].strip()
                    if gen:
                        return gen, None
                if data.get("error"):
                    return None, f"HF API error: {data.get('error')}"
            if isinstance(data, str) and data.strip():
                return data.strip(), None
            return None, "Empty response from Hugging Face Inference API."
    except Exception as e:
        return None, f"Hugging Face error: {e}"


# --- UI ---
def sidebar():
    with st.sidebar:
        st.header("Settings")

        # Model selector
        model = st.selectbox(
            "Model",
            [
                "OpenAI: gpt-3.5-turbo",
                "HuggingFace: mistralai/Mistral-7B-Instruct-v0.2",
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
            # For HF build prompt from history except the last user msg
            history = st.session_state["messages"][:-1]
            new_user = st.session_state["messages"][-1]["content"]
            with st.spinner("Calling Hugging Face…"):
                assistant_text, error = call_huggingface(effective_system_prompt, history, new_user)

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
            elif provider == "huggingface":
                st.caption("Hugging Face Inference request")
                st.write("Model:", raw.get("model"))
                st.write("Parameters:")
                st.json(raw.get("parameters"))
                st.write("Prompt:")
                st.code(raw.get("prompt", ""))
            else:
                st.json(raw)


if __name__ == "__main__":
    main()
