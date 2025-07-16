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
    subject: str  # ✅ added
    short_questions: int
    long_questions: int
    topics: List[str]
    num_assignments: int  # ✅ total marks

@app.post("/generate_assignments")
async def generate_assignments(request: AssignmentRequest):
    short_q_mark = 2
    long_q_mark = 5

    total_marks = request.short_questions * short_q_mark + request.long_questions * long_q_mark

    if total_marks != request.num_assignments:
        raise HTTPException(
            status_code=400,
            detail=f"Total marks from questions = {total_marks}, but expected {request.num_assignments}."
        )

    short_qs = await generate_groq_questions(request.subject, request.topics, request.short_questions, "short", short_q_mark)
    long_qs = await generate_groq_questions(request.subject, request.topics, request.long_questions, "long", long_q_mark)

    return {
        "subject": request.subject,
        "total_marks": total_marks,
        "short_question_marks": request.short_questions * short_q_mark,
        "long_question_marks": request.long_questions * long_q_mark,
        "short_questions": short_qs,
        "long_questions": long_qs
    }

async def generate_groq_questions(subject: str, topics: List[str], count: int, qtype: str, marks: int):
    prompt = f"""
You are an assignment question generator.

Generate {count} {qtype} type questions from the subject: {subject}, and these topics: {', '.join(topics)}.

- Short questions should be definition-based or objective type.
- Long questions should be analytical, asking for explanations or applications.
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
        return [{"question": q, "marks": marks} for q in questions[:count]]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
