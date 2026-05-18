import os
from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session
from supabase import create_client
from engine.db.database import get_db
from engine.models.document import Document
from engine.services.document_service import parse_pdf_from_bytes, chunk_text
from engine.services.embedding_service import embed_and_store
from dotenv import load_dotenv
import tempfile

load_dotenv()

router = APIRouter()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

@router.post("/doc")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):

    # Step 1 - Read file bytes
    file_bytes = await file.read()

    # Step 2 - Upload to Supabase Storage
    storage_path = f"pdfs/{file.filename}"
    supabase.storage.from_("documents").upload(
        path=storage_path,
        file=file_bytes,
        file_options={"content-type": "application/pdf", "upsert": "true"}
    )
    storage_url = supabase.storage.from_("documents").get_public_url(storage_path)

    # Step 3 - Parse PDF from bytes directly (no disk)
    text = parse_pdf_from_bytes(file_bytes)
    chunks = chunk_text(text)

    # Step 4 - Save metadata to PostgreSQL
    doc = Document(filename=file.filename, storage_url=storage_url)
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Step 5 - Embed and store in pgvector
    embed_and_store(chunks, doc.id)

    return {
        "message": "Document uploaded successfully",
        "document_id": doc.id,
        "chunks_created": len(chunks),
        "filename": file.filename,
        "storage_url": storage_url
    }