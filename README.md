# Customer Service Chatbot Sandbox

An interactive Streamlit app to prototype and test customer service chatbot prompts against two backends:

- OpenAI Chat Completions (`gpt-3.5-turbo`)
- Hugging Face Inference API (Mistral: `mistralai/Mistral-7B-Instruct-v0.3`)

The app provides a sidebar for model selection, a live-editable system prompt, reset and example buttons, chat history rendering, and a raw request viewer.

## Quick Start

1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Set environment variables

Copy `.env.example` to `.env` and add your keys, or export them directly in your shell.

```bash
cp .env.example .env
# Then edit .env to add your keys
# OPENAI_API_KEY=...
# HF_API_KEY=...
```

Alternatively, export in your shell session:

```bash
export OPENAI_API_KEY=your_openai_key
export HF_API_KEY=your_hf_key
```

4) Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

## Features

- Sidebar controls:
  - Model selector: `OpenAI: gpt-3.5-turbo` or `HuggingFace: mistralai/Mistral-7B-Instruct-v0.3`
  - Editable System Prompt text area
  - Reset Conversation (clears history, keeps system prompt)
- Load Example Prompt (empathetic, policy-aware customer service template). The example prompt is pre-filled by default on first launch; you can edit or clear it anytime.
- Main area:
  - Chat history rendered using `st.chat_message`
  - `st.chat_input` for live input
  - "Show raw request" expander displaying the exact payload (OpenAI) or prompt + parameters (Hugging Face)
  - Knowledge Base: Upload a PDF in the sidebar under "Knowledge Base (PDF)". The app extracts text, chunks it, and retrieves top matching excerpts for each question, augmenting the system prompt. Use "Clear KB" to remove it.

## Pushing to GitHub

Initialize a repo and push to GitHub:

```bash
git init
git add .
git commit -m "Add Customer Service Chatbot Sandbox"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Deploying on Hugging Face Spaces

1) Create a new Space (Streamlit) and push your repo files there.
2) In the Space Settings, add the following Secrets:
   - `OPENAI_API_KEY`
   - `HF_API_KEY`
3) The app reads these from the environment automatically.

Notes:
- The Hugging Face Inference API may cold-start a model on first request; the app waits for the model when calling.
- Errors and missing keys are surfaced in-app via warnings/errors.

## Project Structure

```
.
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
