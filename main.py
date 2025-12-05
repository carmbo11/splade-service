from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os

app = FastAPI(title="SPLADE Embedding Service")

# CORS for Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy load model to speed up startup
_model = None

def get_model():
    global _model
    if _model is None:
        from fastembed import SparseTextEmbedding
        _model = SparseTextEmbedding("Qdrant/SPLADE_PP_en_v1")
    return _model

class EmbedRequest(BaseModel):
    text: str
    texts: Optional[List[str]] = None

class SparseVector(BaseModel):
    indices: List[int]
    values: List[float]

class EmbedResponse(BaseModel):
    sparse_vector: SparseVector

class BatchEmbedResponse(BaseModel):
    sparse_vectors: List[SparseVector]

@app.get("/")
def health():
    return {"status": "ok", "service": "splade-embeddings"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/embed", response_model=EmbedResponse)
def embed_single(request: EmbedRequest):
    """Generate sparse embedding for a single text"""
    try:
        model = get_model()
        embeddings = list(model.embed([request.text]))

        return EmbedResponse(
            sparse_vector=SparseVector(
                indices=embeddings[0].indices.tolist(),
                values=embeddings[0].values.tolist()
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/batch", response_model=BatchEmbedResponse)
def embed_batch(request: EmbedRequest):
    """Generate sparse embeddings for multiple texts"""
    texts = request.texts or [request.text]

    try:
        model = get_model()
        embeddings = list(model.embed(texts))

        return BatchEmbedResponse(
            sparse_vectors=[
                SparseVector(
                    indices=emb.indices.tolist(),
                    values=emb.values.tolist()
                )
                for emb in embeddings
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
