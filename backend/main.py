from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 🔹 Import your inference function
# This will automatically load models ONCE at startup
from predict import segment_text

app = FastAPI(title="LexiDesk Sentence Detection API")

# -------------------------------------------------
# CORS configuration (Frontend access)
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Request / Response Schemas
# -------------------------------------------------
class SentenceDetectionRequest(BaseModel):
    text: str

class SentenceDetectionResponse(BaseModel):
    sentences: list[str]
    count: int 
# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Hello from your CNN-CRF powered backend!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=SentenceDetectionResponse)
def predict_sentences(request: SentenceDetectionRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    try:
        sentences = segment_text(request.text)
        return {"sentences": sentences, "count": len(sentences)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentence detection failed: {str(e)}")

