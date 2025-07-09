from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize FastAPI
app = FastAPI()

# Request model
class QuarterlySummaryRequest(BaseModel):
    summaries: list[str]  # List of individual summaries

# Route: POST /summarize
@app.post("/summarize")
async def summarize(data: QuarterlySummaryRequest):
    if not data.summaries:
        return {"error": "No summaries provided."}

    # Concatenate all summaries into one block
    combined_input = "\n\n".join(data.summaries)

    # Prompt for generating a team-level quarterly summary
    prompt = (
        "You are an assistant writing a quarterly summary for a business newsletter.\n\n"
        "You will be given multiple individual achievement summaries. Your task is to:\n"
        "- Read them all\n"
        "- Extract key themes, contributions, and patterns\n"
        "- Combine them into a single, unified quarterly summary from a team perspective\n\n"
        "Your output must:\n"
        "- Be written in third-person, focused on the team, not individuals\n"
        "- Use plain text only (no markdown)\n"
        "- Be under 250 words\n"
        "- Use bullet points or labeled sections\n"
        "- Group related items (e.g., Certifications, Go-Lives, Innovations)\n"
        "- Avoid repetition or listing the same type of achievement separately for each person\n"
        "- Never use names. Refer to 'the team', 'team members', or 'the group'\n\n"
        "Only return one final, polished team-level summary. Do not include individual breakdowns or restate the input."
    )

    # Call Groq API
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
                    {"role": "user", "content": combined_input}
                ],
                "temperature": 0.5
            }
        )

        # Error handling
        if response.status_code != 200:
            return {"error": "Groq API error", "details": response.text}

        groq_result = response.json()
        return {"summary": groq_result["choices"][0]["message"]["content"]}
