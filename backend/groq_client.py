import os
from dotenv import load_dotenv
import aiohttp

load_dotenv()

# Always strip to avoid hidden newlines/spaces
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()

if not GROQ_API_KEY:
    raise ValueError("Please set GROQ_API_KEY in .env")

async def groq_generate_async(system_prompt: str, user_prompt: str,
                              max_tokens: int = 512, temperature: float = 0.0) -> str:
    """Async Groq call."""
    url = "https://api.groq.com/openai/v1/chat/completions"   #  correct endpoint
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"Groq API failed: {resp.status} {error_text}")
            result = await resp.json()
            return result["choices"][0]["message"]["content"].strip()
