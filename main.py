from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()

class QuarterlySummaryRequest(BaseModel):
    summaries: list[str]

@app.post("/summarize")
async def summarize(data: QuarterlySummaryRequest):
    combined = "\n\n".join(data.summaries)

    prompt = (
        "You are an assistant writing a high-level quarterly business newsletter summary "
        "based on multiple individual achievement summaries. Your task is to synthesize these "
        "into a polished, readable summary.\n\n"
        "Instructions:\n"
        "- Group common themes (e.g., Go-Lives, Feedback, Achievements)\n"
        "- Highlight key accomplishments and outcomes\n"
        "- Avoid repetition or naming individuals (write as a team-level update)\n"
        "- Use bullet points for readability\n"
        "- Keep the total under 700 words"
    )

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": combined}
                ],
                "temperature": 0.5
            }
        )

    if response.status_code != 200:
        return {"error": "Groq API error", "details": response.text}

    result = response.json()
    return {
        "summary": result["choices"][0]["message"]["content"]
    }


