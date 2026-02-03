from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from dotenv import load_dotenv
import os

# Load environment variables (e.g. GOOGLE_API_KEY)
load_dotenv()

# HARD-CODED API KEY FOR LOCAL DEV ONLY.
# Replace the placeholder string with your real Gemini key and DO NOT commit it.
# Set GOOGLE_API_KEY in your environment or local .env file.
PERSIST_DIR = "db/chroma_db"


def get_vectorstore():
    """Initialise Chroma vector store using existing persisted DB, with Gemini embeddings."""
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )
    return db


db = get_vectorstore()
retriever = db.as_retriever(search_kwargs={"k": 5})
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

app = FastAPI(title="UniBot RAG API")

# Allow your HTML app (served from filesystem / localhost) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RAGQuery(BaseModel):
    question: str


@app.post("/rag-chat")
async def rag_chat(body: RAGQuery):
    """
    Simple RAG endpoint:
    - Receives a question
    - Retrieves relevant chunks from Chroma
    - Generates an answer constrained to those documents
    """
    query = body.question.strip()
    if not query:
        return {"answer": "Please provide a non-empty question."}

    # 1) Retrieve relevant documents
    relevant_docs = retriever.invoke(query)

    if not relevant_docs:
        return {
            "answer": "I couldn't find any relevant information in the documents for that question."
        }

    # 2) Build a combined prompt (pattern based on 3_answer_generation.py)
    combined_input = f"""You are a helpful university assistant. 
Answer the question using ONLY the information from the provided documents.

Question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

If you can't find the answer in the documents, say:
"I don't have enough information to answer that question based on the provided documents."
"""

    messages = [
        SystemMessage(
            content=(
                "You are a helpful university assistant. "
                "Only use the information in the provided documents as your knowledge source. "
                "If the answer is not in the documents, say you don't have enough information."
            )
        ),
        HumanMessage(content=combined_input),
    ]

    result = llm.invoke(messages)
    return {"answer": result.content}


@app.get("/health")
async def health():
    return {"status": "ok"}

