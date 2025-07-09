from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()

# Updated model: accepts a single block of text
class QuarterlySummaryRequest(BaseModel):
    content: str

@app.post("/summarize")
async def summarize(data: QuarterlySummaryRequest):
    if not data.content:
        return {"error": "No content provided."}

    # Prompt for summarizing raw unstructured text
    prompt = (
        "You are an assistant writing a unified quarterly team summary for a business newsletter.\n\n"
        "You will receive a single large block of text containing multiple individual summaries. "
        "These summaries may be mashed together with no clear formatting, line breaks, or structure.\n\n"
        "Your job is to:\n"
        "- Read through all the content\n"
        "- Identify key accomplishments, themes, and repeated patterns\n"
        "- Ignore any repeated phrases or 'Here is a summary' style preambles\n"
        "- Write one clean, professional, 500-word max newsletter summary for the entire team\n\n"
        "The output should:\n"
        "- Use bullet points or short labeled sections\n"
        "- Refer to 'the team' instead of individuals by name\n"
        "- Use plain text only (no markdown)\n"
        "- Only return one clean, final summary"
    )

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": data.content}
                ],
                "temperature": 0.5
            }
        )

        if response.status_code != 200:
            return {"error": "Groq API error", "details": response.text}

        groq_result = response.json()
        return {"summary": groq_result["choices"][0]["message"]["content"]}
