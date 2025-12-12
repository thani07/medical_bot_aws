# main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware

from db import init_db, SessionLocal, create_session, get_sessions, get_session, update_session_title, add_message, get_messages

# Initialize DB (creates tables if necessary)
init_db()

# Groq client config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Use the models you requested
CONV_MODEL = os.getenv("CONV_MODEL", "openai/gpt-oss-120b")
TITLE_MODEL = os.getenv("TITLE_MODEL", "llama-3.3-70b-versatile")

# A concise system prompt (you can extend as needed)
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """
You are an AI medical assistant chatbot.
Answer only medical questions about symptoms, diagnosis, treatment, prevention, medicines, and recovery.
If the user's question is outside the medical domain, reply: "I can only answer medical-related questions."
Be concise and use simple sentences. If listing steps or symptoms, use bullet points.
""".strip())

app = FastAPI(title="Medical Chatbot API")

# Allow cross origin requests (so Lovable AI frontend or local dev can call it)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production to allowed domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Request/Response models ----
class NewSessionResponse(BaseModel):
    session_id: str
    title: str

    class Config:
        schema_extra = {
            "example": {
                "session_id": "a2b93d6d-46f1-4fa3-8dd2-b78508102cf8",
                "title": "New Chat"
            }
        }

class SessionSummary(BaseModel):
    id: str
    title: str
    created_at: str

    class Config:
        schema_extra = {
            "example": {
                "id": "a2b93d6d-46f1-4fa3-8dd2-b78508102cf8",
                "title": "Fever Diagnosis",
                "created_at": "2025-11-25T14:32:05.123Z"
            }
        }


class MessageItem(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    created_at: str

    class Config:
        schema_extra = {
            "example": {
                "id": "427df218-957f-4e50-afd4-e26b9a07c1c7",
                "session_id": "a2b93d6d-46f1-4fa3-8dd2-b78508102cf8",
                "role": "assistant",
                "content": "Fever, headache, chills are common malaria symptoms.",
                "created_at": "2025-11-25T14:33:47.991Z"
            }
        }


class SendMessageRequest(BaseModel):
    session_id: str
    message: str
    model: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "session_id": "9f84c5d1-2a33-4e5b-9411-2419e59bfba0",
                "message": "What are symptoms of malaria?",
                "model": "openai/gpt-oss-120b"
            }
        }


class SendMessageResponse(BaseModel):
    assistant: str
    session_id: str

    class Config:
        schema_extra = {
            "example": {
                "assistant": "Malaria symptoms include fever, chills, headache and fatigue.",
                "session_id": "a2b93d6d-46f1-4fa3-8dd2-b78508102cf8"
            }
        }


# Helper LLM calls
def call_llm_for_reply(user_message: str, model: str = None) -> str:
    model_to_use = model or CONV_MODEL
    completion = client.chat.completions.create(
        model=model_to_use,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        top_p=1,
        max_completion_tokens=2048,
        stream=False
    )
    # Groq returns a ChatCompletionMessage object inside choices
    text = completion.choices[0].message.content
    return text.strip()

def call_llm_for_title(first_user_message: str) -> str:
    prompt_system = (
        "You are a short title generator. Produce a very short (2-4 word) descriptive title "
        "for the medical conversation. Reply with the title only."
    )
    completion = client.chat.completions.create(
        model=TITLE_MODEL,
        messages=[
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": f'Conversation first message: "{first_user_message}"'}
        ],
        temperature=0.2,
        top_p=1,
        max_completion_tokens=16,
        stream=False
    )
    title = completion.choices[0].message.content.strip()
    title = title.split("\n")[0].strip()
    # keep max 4 words
    words = title.split()
    if len(words) > 4:
        title = " ".join(words[:4])
    return title

# ---- API endpoints ----
@app.post("/new_session", response_model=NewSessionResponse)
def api_new_session():
    db = SessionLocal()
    try:
        sess = create_session(db, title="New Chat")
        # add assistant welcome message
        welcome = "👋 Hello! I'm your AI medical assistant. Ask me about symptoms, recovery, or health tips."
        add_message(db, sess.id, "assistant", welcome)
        return {"session_id": sess.id, "title": sess.title}
    finally:
        db.close()

@app.get("/sessions", response_model=List[SessionSummary])
def api_sessions():
    db = SessionLocal()
    try:
        rows = get_sessions(db)
        return [{"id": r.id, "title": r.title, "created_at": r.created_at.isoformat()} for r in rows]
    finally:
        db.close()

@app.get("/messages/{session_id}", response_model=List[MessageItem])
def api_messages(session_id: str):
    db = SessionLocal()
    try:
        sess = get_session(db, session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        msgs = get_messages(db, session_id)
        return [
            {"id": m.id, "session_id": m.session_id, "role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
            for m in msgs
        ]
    finally:
        db.close()

@app.post("/send_message", response_model=SendMessageResponse)
def api_send_message(payload: SendMessageRequest):
    db = SessionLocal()
    try:
        sess = get_session(db, payload.session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        # Save user message
        add_message(db, payload.session_id, "user", payload.message)
        # Call LLM
        assistant_text = call_llm_for_reply(payload.message, model=payload.model)
        # Save assistant message
        add_message(db, payload.session_id, "assistant", assistant_text)
        # If session title is default "New Chat", generate a short title from first user message
        if not sess.title or sess.title.strip().lower() == "new chat":
            # find first user message in this session
            msgs = get_messages(db, payload.session_id)
            first_user = None
            for m in msgs:
                if m.role == "user":
                    first_user = m.content
                    break
            if first_user:
                try:
                    title = call_llm_for_title(first_user)
                    if title:
                        update_session_title(db, payload.session_id, title)
                except Exception:
                    # ignore title gen errors
                    pass
        return {"assistant": assistant_text, "session_id": payload.session_id}
    finally:
        db.close()
