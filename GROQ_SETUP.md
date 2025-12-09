# Groq API Setup

This app now supports Groq's free API for blazing-fast inference with smaller language models. Groq is perfect for demonstrating hallucination patterns in smaller LLMs compared to GPT-3.5.

## Why Groq?

- **Free tier**: 30 requests/minute, 14,400 requests/day
- **Extremely fast**: Sub-second response times
- **No credit card required**
- **Works with Streamlit Cloud**: Perfect for remote deployment
- **Smaller model**: Llama 3.2 1B (1 billion parameters vs GPT-3.5's 175B)

## Getting Your Free Groq API Key

1. **Sign up for a free Groq account**
   - Go to https://console.groq.com
   - Click "Sign Up" and create a free account

2. **Generate an API key**
   - After logging in, go to https://console.groq.com/keys
   - Click "Create API Key"
   - Give it a name (e.g., "chatbot-app")
   - Copy your API key (starts with `gsk_...`)

3. **Add the key to your app**
   - **For local development**:
     - Create a `.env` file in this directory (copy from `.env.example`)
     - Add the line: `GROQ_API_KEY=gsk_your_key_here`
   - **For Streamlit Cloud**:
     - Go to your app settings on Streamlit Cloud
     - Find the "Secrets" section
     - Add: `GROQ_API_KEY = "gsk_your_key_here"`

## Free Tier Limits

- **Rate limit**: 30 requests per minute
- **Daily limit**: 14,400 requests per day
- **No cost**: Completely free
- **Model**: Uses `llama-3.2-1b-preview` - a 1B parameter model
  - Much smaller than GPT-3.5 (175B parameters)
  - More prone to errors and hallucinations
  - Perfect for teaching about LLM limitations!

## Model Comparison

| Model | Parameters | Provider | Hallucination Risk | Speed | Use Case |
|-------|-----------|----------|-------------------|-------|----------|
| Llama 3.2 1B | 1B | Groq | High | Very Fast | Teaching/Demo |
| GPT-3.5-Turbo | 175B | OpenAI | Low | Fast | Production Use |

## Why Llama 3.2 1B is Perfect for Your Bootcamp

1. **Shows limitations clearly**: The 1B model will make more mistakes, helping learners understand the importance of model size
2. **Fast iteration**: Sub-second responses mean students can try many examples quickly
3. **Free and reliable**: No risk of hitting expensive API limits during class
4. **Real comparison**: Students can switch between models instantly and see the difference

## Deploying to Streamlit Cloud

When deploying, add your `GROQ_API_KEY` to Streamlit Cloud secrets:
1. Go to your app settings on Streamlit Cloud
2. Find the "Secrets" section
3. Add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
4. Save and your app will automatically restart
