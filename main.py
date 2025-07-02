from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv

# Load Groq API key from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Define FastAPI app
app = FastAPI()

# Pydantic model
class SummaryRequest(BaseModel):
    name: str
    go_lives: str = ""
    deals: str = ""
    achievements: str = ""
    feedback: str = ""
    scoe: str = ""
    core_usecases: str = ""
    fusion_usecases: str = ""
    innovations: str = ""
    issues: str = ""
    collaborations: str = ""

@app.post("/summarize")
async def summarize(data: SummaryRequest):
    # Build content block with non-empty fields
    lines = [f"Name: {data.name}"]
    for field, value in data.dict().items():
        if field != "name" and value.strip():
            lines.append(f"{field.replace('_', ' ').title()}: {value.strip()}")
    content = "\n".join(lines)

    prompt = (
        "You are an assistant writing polished business newsletter summaries "
        "for quarterly *individual* achievements based on structured field-wise input. "
        "The input will contain labeled fields (e.g., Go-Lives, Feedback, etc.).\n\n"
        "Generate a summary that:\n"
        "- Uses bullet points\n"
        "- Includes the field name in bold before each point (e.g., **Go-Lives:**)\n"
        "- Skips empty or irrelevant fields\n"
        "- Writes from the perspective of the individual (not 'our team')\n"
        "- Keeps the total length under 300 words"
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
                    {"role": "user", "content": content}
                ],
                "temperature": 0.5
            }
        )

    if response.status_code != 200:
        return {
            "error": "Groq API error",
            "details": response.text
        }

    result = response.json()
    return {
        "summary": result["choices"][0]["message"]["content"]
    }
