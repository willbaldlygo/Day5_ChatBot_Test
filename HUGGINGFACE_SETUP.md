# Hugging Face API Setup

This app now supports Hugging Face's free Inference API to access smaller language models that are great for demonstrating hallucination patterns in older/smaller LLMs.

## Getting Your Free Hugging Face API Key

1. **Sign up for a free Hugging Face account**
   - Go to https://huggingface.co/join
   - Create a free account (no credit card required)

2. **Generate an API token**
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Give it a name (e.g., "chatbot-app")
   - Select "Read" access (default)
   - Click "Generate"
   - Copy your token (starts with `hf_...`)

3. **Add the token to your app**
   - Create a `.env` file in this directory (copy from `.env.example`)
   - Add the line: `HUGGINGFACE_API_KEY=hf_your_token_here`
   - Or add it to Streamlit Cloud secrets when deploying

## Free Tier Limits

- **Rate limit**: ~1000 requests per hour
- **No cost**: Completely free
- **Model**: Uses `google/flan-t5-small` - a 80M parameter model
  - Much smaller than GPT-3.5 (175B parameters)
  - More prone to hallucination and errors
  - Perfect for teaching about LLM limitations!

## First Request Note

The first time you use a model, Hugging Face needs to "warm up" the model, which can take 20-30 seconds. Subsequent requests are fast. If you see a "Model is loading" message, just wait and try again.

## Model Comparison

| Model | Parameters | Hallucination Risk | Use Case |
|-------|-----------|-------------------|----------|
| FLAN-T5-Small | 80M | High | Teaching/Demo |
| GPT-3.5-Turbo | 175B | Low | Production Use |

## Deploying to Streamlit Cloud

When deploying, add your `HUGGINGFACE_API_KEY` to Streamlit Cloud secrets:
1. Go to your app settings on Streamlit Cloud
2. Find the "Secrets" section
3. Add: `HUGGINGFACE_API_KEY = "hf_your_token_here"`
