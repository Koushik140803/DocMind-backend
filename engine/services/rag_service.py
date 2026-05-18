from engine.services.embedding_service import search_similar_chunks

def build_prompt(query: str, chunks: list[str], history: list[dict] = []) -> str:
    context = "\n\n".join(chunks)

    history_text = ""
    if history:
        history_text = "\n\nConversation so far:\n"
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}
{history_text}
Current question: {query}

Answer:"""

    return prompt