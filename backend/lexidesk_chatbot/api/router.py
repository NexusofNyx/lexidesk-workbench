from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from lexidesk_chatbot.retrieval.retriever import (
    retrieve,
    generate_answer
)

router = APIRouter(prefix="/chat", tags=["Chatbot"])

# --------------------------------------------------
# Schemas
# --------------------------------------------------

class QARequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class Source(BaseModel):
    text: str
    metadata: Dict[str, Any]
    score: float


class QAResponse(BaseModel):
    answer: str
    sources: List[Source]


# --------------------------------------------------
# POST /chat
# --------------------------------------------------

@router.post("", response_model=QAResponse)
def chat(req: QARequest):
    # ------------------------------
    # 0. Validate input
    # ------------------------------
    if not req.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    # ------------------------------
    # 1. Retrieve chunks
    # ------------------------------
    try:
        raw_chunks = retrieve(
            query=req.question,
            top_k=req.top_k or 5
        )
    except Exception as e:
        print("❌ Retrieval error:", e)
        raise HTTPException(
            status_code=500,
            detail="Vector retrieval failed"
        )

    if not raw_chunks:
        return QAResponse(
            answer="The provided documents do not contain this information.",
            sources=[]
        )

    # ------------------------------
    # 2. Normalize chunks
    # ------------------------------
    sources: List[Source] = []
    gemini_chunks: List[Dict[str, Any]] = []

    for chunk in raw_chunks:
        try:
            score = float(chunk.get("score", chunk.get("distance", 0.0)))

            metadata = chunk.get("metadata") or {}

            sources.append(
                Source(
                    text=chunk["text"],
                    metadata=metadata,
                    score=score
                )
            )

            # 🔑 Gemini MUST receive dicts, not Pydantic objects
            gemini_chunks.append({
                "text": chunk["text"],
                "metadata": metadata
            })

        except Exception as e:
            print("⚠️ Skipping malformed chunk:", chunk, e)

    if not gemini_chunks:
        return QAResponse(
            answer="The provided documents do not contain this information.",
            sources=[]
        )

    # ------------------------------
    # 3. Generate Answer
    # ------------------------------
    try:
        answer = generate_answer(
            query=req.question,
            retrieved_chunks=gemini_chunks
        )
    except Exception as e:
        print("❌ Gemini error:", e)
        raise HTTPException(
            status_code=500,
            detail="Answer generation failed"
        )

    # ------------------------------
    # 4. Response
    # ------------------------------
    return QAResponse(
        answer=answer,
        sources=sources
    )
