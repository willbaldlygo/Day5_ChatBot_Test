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

APP_TITLE = "Digital Peninsula: ChatBot Sandbox"


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
        # Supported Groq default; you can switch to HF in the sidebar
        st.session_state["model_choice"] = "Groq: llama-3.1-8b-instant"
    if "raw_request" not in st.session_state:
        st.session_state["raw_request"] = None
    if "kb" not in st.session_state:
        st.session_state["kb"] = None  # dict with {text, chunks, pages, filename}
    if "hf_model_id" not in st.session_state:
        # Default HF model (you can edit in the sidebar)
        st.session_state["hf_model_id"] = "mistralai/Mistral-7B-Instruct-v0.2"
    if "speculative_mode" not in st.session_state:
        st.session_state["speculative_mode"] = False


def load_example_system_prompt() -> str:
    return (
        "You are a helpful AI assistant. Your job is to answer questions as clearly "
        "and accurately as you can. Keep your tone friendly, but professional."
    )


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


# --- Groq helpers: live model list + label formatting ---
@st.cache_data(ttl=600)
def fetch_groq_models() -> List[str]:
    """Return list of Groq model IDs from the live /models endpoint, or [] on failure."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return []
    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json() or {}
        items = data.get("data", [])
        ids = [m.get("id", "") for m in items if isinstance(m, dict)]
        # Prefer stable options first if present
        preferred_order = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ]
        ordered = [m for m in preferred_order if m in ids] + sorted([m for m in ids if m not in preferred_order])
        return ordered
    except Exception:
        return []


def label_for_groq(model_id: str) -> str:
    return f"Groq: {model_id}"


def parse_model_choice(choice: str) -> Tuple[str, str]:
    """Return (provider, model_id) from a selectbox label."""
    if choice.startswith("Groq: "):
        return "groq", choice.split("Groq: ", 1)[1]
    if choice.startswith("OpenAI: "):
        return "openai", choice.split("OpenAI: ", 1)[1]
    if choice.startswith("HuggingFace: "):
        return "huggingface", choice.split("HuggingFace: ", 1)[1]
    return "unknown", choice


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


def call_groq(messages: List[Dict[str, str]], model_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Call Groq Chat Completions API for the given model_id.

    Returns (assistant_text, error_message). Also stores raw request in session state.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None, "Missing GROQ_API_KEY. Set it in your environment or .env."

    temperature = 0.3
    max_tokens = 512

    payload = {
        "model": model_id,
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
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            return None, f"Groq API error {resp.status_code}: {resp.text}"
        data = resp.json()
        text = None
        if isinstance(data, dict) and data.get("choices"):
            choice = data["choices"][0]
            msg = choice.get("message") if isinstance(choice, dict) else None
            if msg and isinstance(msg.get("content"), str):
                text = msg["content"]
        if not text or not text.strip():
            return None, "Empty response from Groq. (Model may be unavailable or deprecated.)"
        return text.strip(), None
    except requests.Timeout:
        return None, "Groq error: request timed out. Try a different model or reduce max_tokens."
    except Exception as e:
        return None, f"Groq error: {e}"


def call_hf_inference(messages: List[Dict[str, str]], model_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Hugging Face Inference API backend (text-generation).
    Expects HF_TOKEN in env or Streamlit secrets.
    We map OpenAI-style messages to a single prompt appropriate for chat-tuned models.
    """
    hf_token = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN", None)
    if not hf_token:
        return None, "Missing HF_TOKEN. Create a Hugging Face access token and set HF_TOKEN."

    # Collapse messages into a simple chat prompt
    def to_prompt(msgs):
        lines = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                lines.append(f"[SYSTEM] {content}")
            elif role == "assistant":
                lines.append(f"[ASSISTANT] {content}")
            else:
                lines.append(f"[USER] {content}")
        lines.append("[ASSISTANT]")
        return "\n".join(lines)

    prompt = to_prompt(messages)

    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.9,          # a bit higher to encourage “confident improv”
            "top_p": 0.95,
            "repetition_penalty": 1.05,
            "return_full_text": False
        },
        "options": {
            "wait_for_model": True,      # first call cold start
            "use_cache": True
        }
    }

    st.session_state["raw_request"] = {"provider": "huggingface", "payload": {"model": model_id, **payload}}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        if r.status_code != 200:
            # HF returns informative errors (e.g., model loading, rate limit)
            return None, f"HF API error {r.status_code}: {r.text[:500]}"
        data = r.json()
        # Typical shape: [{"generated_text": "..."}]
        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            text = data[0]["generated_text"]
            return text.strip(), None
        # Some models return a dict with error
        if isinstance(data, dict) and "error" in data:
            return None, f"HF API: {data['error']}"
        return None, "Empty or unrecognised response from HF Inference API."
    except requests.Timeout:
        return None, "HF Inference API timeout."
    except Exception as e:
        return None, f"HF error: {e}"


# --- UI ---
def sidebar():
    with st.sidebar:
        st.header("Settings")

        # --- Dynamic Groq model list (falls back if API/key missing) ---
        groq_models = fetch_groq_models()
        groq_default = "llama-3.1-8b-instant"
        groq_labels = [label_for_groq(m) for m in groq_models] if groq_models else [label_for_groq(groq_default)]

        # --- Hugging Face model entry (free text so you can use any HF model ID) ---
        st.subheader("Hugging Face")
        hf_model_id = st.text_input(
            "HF model repo (owner/name)",
            value=st.session_state.get("hf_model_id", "mistralai/Mistral-7B-Instruct-v0.2"),
            help="Examples: mistralai/Mistral-7B-Instruct-v0.2, mistralai/Mixtral-8x7B-Instruct-v0.1",
        )
        # Persist selection
        st.session_state["hf_model_id"] = hf_model_id.strip()

        # --- Model selector (Groq, OpenAI, and HF) ---
        model_options = (
            groq_labels
            + ["OpenAI: gpt-3.5-turbo"]
            + [f"HuggingFace: {st.session_state['hf_model_id']}"]
        )
        current_choice = st.session_state.get("model_choice", model_options[0])
        if current_choice not in model_options:
            current_choice = model_options[0]
        default_index = model_options.index(current_choice)

        model = st.selectbox(
            "Model",
            model_options,
            index=default_index,
            help="Groq list comes from the live /models endpoint. Hugging Face uses the repo you typed above.",
        )
        st.session_state["model_choice"] = model

        # Speculative mode toggle (lets models improvise beyond their cutoffs)
        st.checkbox(
            "Speculative Mode (improvise confidently)",
            value=st.session_state.get("speculative_mode", False),
            key="speculative_mode",
            help="When on, the system prompt encourages imaginative, confident answers even when data may be unknown.",
        )

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
        hf_ok = bool(os.getenv("HF_TOKEN"))
        st.caption(f"OpenAI: {'✅ set' if openai_ok else '⚠️ missing'}")
        st.caption(f"Groq: {'✅ set' if groq_ok else '⚠️ missing'}")
        st.caption(f"Hugging Face: {'✅ set' if hf_ok else '⚠️ missing'}")

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
            # Groq check
            if st.button("Test Groq Connectivity"):
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    st.error("GROQ_API_KEY is missing. Add it to your environment or Secrets.")
                else:
                    provider, model_id = parse_model_choice(st.session_state.get("model_choice", ""))
                    if provider != "groq":
                        st.info("Switch the Model selector to a Groq model first.")
                    else:
                        url = "https://api.groq.com/openai/v1/chat/completions"
                        payload = {
                            "model": model_id,
                            "messages": [
                                {"role": "system", "content": "Connectivity test"},
                                {"role": "user", "content": "Say 'hello'"},
                            ],
                            "max_tokens": 8,
                            "temperature": 0.0,
                        }
                        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}
                        try:
                            r = requests.post(url, headers=headers, json=payload, timeout=30)
                            if r.status_code != 200:
                                st.error(f"Groq API error {r.status_code}: {r.text[:500]}")
                            else:
                                st.success("Groq connectivity OK")
                                try:
                                    st.json(r.json())
                                except Exception:
                                    st.write(r.text)
                        except requests.Timeout:
                            st.error("Groq connectivity test timed out.")
                        except Exception as e:
                            st.error(f"Groq connectivity test failed: {e}")

            # HF check
            if st.button("Test Hugging Face Connectivity"):
                hf_token = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN", None)
                model_id = st.session_state.get("hf_model_id", "").strip()
                if not hf_token:
                    st.error("HF_TOKEN is missing. Add it to your environment or Secrets.")
                elif not model_id:
                    st.error("Provide a Hugging Face model repo ID above.")
                else:
                    try:
                        url = f"https://api-inference.huggingface.co/models/{model_id}"
                        headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
                        payload = {
                            "inputs": "[SYSTEM] Connectivity test\n[USER] Say 'hello'\n[ASSISTANT]",
                            "parameters": {"max_new_tokens": 12, "temperature": 0.2, "return_full_text": False},
                            "options": {"wait_for_model": True}
                        }
                        r = requests.post(url, headers=headers, json=payload, timeout=120)
                        if r.status_code != 200:
                            st.error(f"HF API error {r.status_code}: {r.text[:500]}")
                        else:
                            st.success("Hugging Face connectivity OK")
                            try:
                                st.json(r.json())
                            except Exception:
                                st.write(r.text)
                    except requests.Timeout:
                        st.error("HF connectivity test timed out.")
                    except Exception as e:
                        st.error(f"HF connectivity test failed: {e}")


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

        # Speculative overlay (lets models improvise confidently)
        if st.session_state.get("speculative_mode"):
            base_system_prompt = (
                "You are a speculative narrator. Invent plausible future details with confidence. "
                "Do not refuse for lack of data; improvise vividly while keeping answers coherent.\n\n"
            ) + base_system_prompt

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

        # Build OpenAI-compatible messages
        api_messages = [{"role": "system", "content": effective_system_prompt}] + st.session_state["messages"]

        provider, model_id = parse_model_choice(model_choice)

        if provider == "openai":
            with st.spinner("Calling OpenAI…"):
                assistant_text, error = call_openai(api_messages)
        elif provider == "groq":
            with st.spinner(f"Calling Groq ({model_id})…"):
                assistant_text, error = call_groq(api_messages, model_id=model_id)
        elif provider == "huggingface":
            # model_id comes from the selectbox label, which mirrors the text_input
            with st.spinner(f"Calling Hugging Face ({model_id})…"):
                assistant_text, error = call_hf_inference(api_messages, model_id=model_id)
        else:
            assistant_text, error = None, "Unknown provider/model selection."

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
            elif provider == "huggingface":
                st.caption("Hugging Face Inference API payload")
                st.json(raw.get("payload"))
            else:
                st.json(raw)


if __name__ == "__main__":
    main()
