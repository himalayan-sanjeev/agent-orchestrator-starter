from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel('gemini-2.0-flash')

@app.get("/api/ai_response")
async def ai_response():
    prompt = "Give a short motivational message for a developer learning AI Agents."

    response = model.generate_content(prompt)

    ai_message = response.text.strip()

    return {"message": ai_message}