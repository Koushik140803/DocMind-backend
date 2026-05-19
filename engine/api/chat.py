from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from engine.services.embedding_service import search_similar_chunks
from engine.services.rag_service import build_prompt
import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import select
from engine.db.database import SessionLocal
from engine.models.conversation import Conversation

load_dotenv()

router = APIRouter()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class ChatRequest(BaseModel):
    question: str
    document_id: int
    session_id: str

def load_history(session_id: str, document_id: int) -> list[dict]:
    db: Session = SessionLocal()
    try:
        results = db.scalars(
            select(Conversation)
            .where(
                Conversation.session_id == session_id,
                Conversation.document_id == document_id
            )
            .order_by(Conversation.created_at.asc())
            .limit(6)
        ).all()
        return [{"role": msg.role, "content": msg.content} for msg in results]
    finally:
        db.close()

def save_message(session_id: str, document_id: int, role: str, content: str):
    db: Session = SessionLocal()
    try:
        msg = Conversation(
            session_id=session_id,
            document_id=document_id,
            role=role,
            content=content
        )
        db.add(msg)
        db.commit()
    finally:
        db.close()

def stream_groq(prompt: str, session_id: str, document_id: int, question: str):
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    full_response = ""
    for chunk in response:
        token = chunk.choices[0].delta.content
        if token:
            full_response += token
            yield token

    save_message(session_id, document_id, "user", question)
    save_message(session_id, document_id, "assistant", full_response)

@router.post("/chat")
async def chat(request: ChatRequest):
    history = load_history(request.session_id, request.document_id)
    chunks = search_similar_chunks(request.question, request.document_id)
    prompt = build_prompt(request.question, chunks, history)
    return StreamingResponse(
        stream_groq(prompt, request.session_id, request.document_id, request.question),
        media_type="text/plain"
    )