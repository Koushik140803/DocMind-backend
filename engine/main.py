from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from engine.db.database import engine as db_engine, Base
from engine.api import documents, chat
import engine.models.document
import engine.models.chunk
import engine.models.conversation

Base.metadata.create_all(bind=db_engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)