import os
import requests
from sqlalchemy.orm import Session
from sqlalchemy import select, text
from engine.db.database import SessionLocal
from engine.models.chunk import Chunk
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

def get_embedding(texts: list[str]) -> list[list[float]]:
    response = requests.post(
        HF_URL,
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        json={"inputs": texts, "options": {"wait_for_model": True}},
        timeout=30
    )
    
    print(f"HF Status: {response.status_code}")
    print(f"HF Response: {response.text[:300]}")
    
    if response.status_code != 200:
        raise Exception(f"HuggingFace API failed: {response.status_code} - {response.text}")
    
    if not response.text.strip():
        raise Exception("HuggingFace returned empty response - model may be loading, retry")
    
    return response.json()

def embed_and_store(chunks: list[str], doc_id: int):
    db: Session = SessionLocal()
    try:
        embeddings = get_embedding(chunks)
        for content, embedding in zip(chunks, embeddings):
            chunk = Chunk(
                document_id=doc_id,
                content=content,
                embedding=embedding
            )
            db.add(chunk)
        db.commit()
    finally:
        db.close()



def search_similar_chunks(query: str, doc_id: int, top_k: int = 5, rrf_k: int = 60) -> list[str]:
    db: Session = SessionLocal()
    try:
        query_embedding = get_embedding([query])[0]
        sql = text("""
            SELECT id, content, score
            FROM hybrid_search(:query_text, CAST(:query_embedding AS vector), :doc_id, :top_k, :rrf_k)
        """)
        results = db.execute(sql, {
            "query_text": query,
            "query_embedding": str(query_embedding),
            "doc_id": doc_id,
            "top_k": top_k,
            "rrf_k": rrf_k
        }).fetchall()
        return [row.content for row in results]
    finally:
        db.close()