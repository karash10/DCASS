# src/api/server.py
"""
DCASS FastAPI Backend.

Exposes the DCASS engine (encode, decode, search, status, benchmark)
as a REST API for the Next.js frontend.

Usage:
    uvicorn src.api.server:app --reload --port 8000
"""

from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DCASS API",
    description="Dynamic Context-Aware Semantic Steganography",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy engine singletons
# ---------------------------------------------------------------------------
_encoder = None
_decoder = None
_initializing = False
_ready = False


def _get_encoder():
    global _encoder
    if _encoder is None:
        from src.engine.encoder import SemanticEncoder
        _encoder = SemanticEncoder(expand_synonyms=True)
        _encoder.load()
    return _encoder


def _get_decoder():
    global _decoder
    if _decoder is None:
        from src.engine.decoder import SemanticDecoder
        _decoder = SemanticDecoder()
        _decoder.load()
    return _decoder


def warmup():
    """Pre-load encoder and decoder on startup."""
    global _initializing, _ready
    _initializing = True
    print("\n" + "=" * 70)
    print("🔥 Warming up DCASS engine...")
    print("=" * 70)
    try:
        # Load encoder (this triggers CLIP model loading)
        print("\n📦 Loading encoder and CLIP model...")
        encoder = _get_encoder()
        print(f"✅ Encoder ready: {encoder}")
        
        # Load decoder
        print("\n📦 Loading decoder...")
        decoder = _get_decoder()
        print(f"✅ Decoder ready: {decoder}")
        
        _ready = True
        print("\n" + "=" * 70)
        print("✅ DCASS engine ready!")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"\n❌ Warmup failed: {e}")
        print("=" * 70 + "\n")
        _ready = False
    finally:
        _initializing = False


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class EncodeRequest(BaseModel):
    message: str
    mode: Literal["best", "round_robin", "balanced"] = "best"
    modalities: list[str] = Field(default=["image", "text", "audio"])


class EncodeResponse(BaseModel):
    media_ids: list[str]
    chunks: list[str]
    encoded: list[dict]
    modality_breakdown: dict[str, int]
    elapsed_ms: float


class DecodeRequest(BaseModel):
    media_ids: list[str]


class DecodeResponse(BaseModel):
    reconstructed_meaning: str
    items: list[dict]
    verification_rate: float
    all_verified: bool
    elapsed_ms: float


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    modalities: list[str] = Field(default=["image", "text", "audio"])


class SearchResponse(BaseModel):
    results: list[dict]
    elapsed_ms: float


class StatusResponse(BaseModel):
    indices: dict
    total_items: int
    device: str
    stealth_models: dict


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/encode", response_model=EncodeResponse)
def encode(req: EncodeRequest):
    t0 = time.perf_counter()
    try:
        encoder = _get_encoder()
        result = encoder.encode(req.message, modalities=req.modalities, diversity_mode=req.mode)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    encoded_items = []
    for enc in result.encoded:
        encoded_items.append({
            "media_id": enc.media.id,
            "modality": enc.media.modality,
            "score": round(enc.media.normalized_score, 4),
            "content": enc.media.content[:120],
        })

    return EncodeResponse(
        media_ids=result.media_ids,
        chunks=[c.original for c in result.chunks],
        encoded=encoded_items,
        modality_breakdown=result.modality_breakdown,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


@app.post("/api/decode", response_model=DecodeResponse)
def decode(req: DecodeRequest):
    t0 = time.perf_counter()
    decoder = _get_decoder()
    result = decoder.decode(req.media_ids)

    items = []
    for d in result.decoded:
        items.append({
            "media_id": d.media_id,
            "modality": d.modality,
            "content": d.content[:200],
            "verified": d.verified,
        })

    return DecodeResponse(
        reconstructed_meaning=result.reconstructed_meaning,
        items=items,
        verification_rate=result.verification_rate,
        all_verified=result.all_verified,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    t0 = time.perf_counter()
    encoder = _get_encoder()
    results = encoder.index.search(req.query, k=req.k, modalities=req.modalities)

    items = []
    for r in results:
        items.append({
            "id": r.id,
            "modality": r.modality,
            "score": round(r.normalized_score, 4),
            "content": r.content[:200],
        })

    return SearchResponse(
        results=items,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


@app.get("/api/status", response_model=StatusResponse)
def status():
    import torch
    indices_path = Path(__file__).parent.parent.parent / "data" / "indices"
    models_path = Path(__file__).parent.parent.parent / "models"

    index_info = {}
    total = 0
    for mod in ["image", "text", "audio"]:
        idx_file = indices_path / f"{mod}.index"
        meta_file = indices_path / f"{mod}_metadata.json"
        if idx_file.exists() and meta_file.exists():
            try:
                import faiss
                idx = faiss.read_index(str(idx_file))
                count = idx.ntotal
                total += count
                index_info[mod] = {"status": "ok", "count": count}
            except Exception as e:
                index_info[mod] = {"status": "error", "error": str(e)}
        else:
            index_info[mod] = {"status": "missing"}

    stealth = {
        "gan_checkpoint": (models_path / "gan" / "final.pt").exists(),
        "rl_checkpoint": (models_path / "rl" / "ppo_agent_final.pt").exists(),
    }

    return StatusResponse(
        indices=index_info,
        total_items=total,
        device="cuda" if torch.cuda.is_available() else "cpu",
        stealth_models=stealth,
    )


@app.get("/api/ready")
def ready():
    """Check if the server is ready to process requests."""
    return {
        "ready": _ready,
        "initializing": _initializing,
        "encoder_loaded": _encoder is not None,
        "decoder_loaded": _decoder is not None,
    }


@app.get("/api/benchmark/latest")
def benchmark_latest():
    results_dir = Path(__file__).parent.parent.parent / "data" / "benchmarks" / "results"
    if not results_dir.exists():
        return {"available": False}

    files = sorted(results_dir.glob("benchmark_*.json"), reverse=True)
    if not files:
        return {"available": False}

    with open(files[0], "r", encoding="utf-8") as f:
        data = json.load(f)

    return {"available": True, "filename": files[0].name, "data": data}
