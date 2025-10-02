from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"
MODEL_NAME = "llama-3.3-70b-versatile"

app = FastAPI()

class AssignmentRequest(BaseModel):
    subject: str
    short_questions: int
    topics: List[str]

@app.post("/generate_assignments")
async def generate_assignments(request: AssignmentRequest):
    # Define mark weight
    short_q_mark = 2

    # ✅ Automatically calculate total marks
    total_marks = request.short_questions * short_q_mark

    # Generate short questions
    short_qs = await generate_groq_questions(request.subject, request.topics, request.short_questions)

    return {
        "subject": request.subject,
        "total_marks": total_marks,
        "short_questions": short_qs
    }

async def generate_groq_questions(subject: str, topics: List[str], count: int):
    prompt = f"""
You are an assignment question generator.

Generate {count} short type questions for the subject "{subject}" from these topics: {', '.join(topics)}.

- Short questions should be definition-based or objective type.
Return only the questions in a list format.
"""
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = response['choices'][0]['message']['content']
        questions = [q.strip("-• ") for q in content.split("\n") if q.strip()]
        return questions[:count]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
