"""FastAPI service exposing REST endpoints for query suggestions,
dynamic word insertion, query recording, statistics, and benchmark execution.
"""

import os
import time
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Query, Response, status, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from autocomplete_engine.core.engine import AutocompleteEngine
from autocomplete_engine.benchmarks.datasets import MultilingualDatasetGenerator
from autocomplete_engine.benchmarks.benchmark_runner import BenchmarkRunner


app = FastAPI(
    title="Multilingual Autocomplete & Query Suggestion Engine",
    description=(
        "Edge-compressed Radix Tree autocomplete engine with "
        "Top-K branch-and-bound pruning and typo correction."
    ),
    version="0.1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global engine instance initialized with realistic multilingual vocabulary
engine = AutocompleteEngine()


# Pre-seed engine on startup
def _seed_initial_vocabulary():
    gen = MultilingualDatasetGenerator(seed=101)
    initial_vocab = gen.generate_vocabulary(total_words=5000)
    engine.batch_load(initial_vocab)

    print(
        f"[+] AutocompleteEngine initialized with "
        f"{len(initial_vocab)} multilingual words."
    )


_seed_initial_vocabulary()


# Pydantic Request Models
class InsertRequest(BaseModel):
    word: str = Field(
        ...,
        min_length=1,
        description="Word or phrase to insert"
    )
    score: float = Field(
        default=1.0,
        ge=0.0,
        description="Initial frequency score"
    )
    script: Optional[str] = Field(
        default=None,
        description="Optional script override"
    )


class RecordQueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Query string executed"
    )
    delta: float = Field(
        default=1.0,
        gt=0.0,
        description="Score increment"
    )
    script: Optional[str] = Field(
        default=None,
        description="Optional script override"
    )


class DeleteRequest(BaseModel):
    word: str = Field(
        ...,
        min_length=1,
        description="Word to tombstone"
    )
    script: Optional[str] = Field(
        default=None,
        description="Optional script override"
    )


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serves the minimal single-page demo interface."""

    html_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "web",
        "index.html"
    )

    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    return HTMLResponse(
        "<h3>Demo UI not found.</h3>",
        status_code=404
    )


@app.get("/api/suggest")
async def suggest(
    response: Response,
    q: str = Query(
        ...,
        description="Query prefix to complete"
    ),
    k: int = Query(
        5,
        ge=1,
        le=50,
        description="Max suggestions to return"
    ),
    fuzzy: bool = Query(
        True,
        description="Enable Levenshtein typo correction"
    ),
    lang: Optional[str] = Query(
        None,
        description=(
            "Optional script override "
            "(latin, telugu, tamil, devanagari, other)"
        )
    )
):
    """Returns Top-K suggestions with typo tolerance and server process time header."""

    t0 = time.perf_counter_ns()

    try:
        results = engine.search(
            prefix=q,
            k=k,
            fuzzy=fuzzy,
            script_override=lang
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )

    elapsed_ms = (
        time.perf_counter_ns() - t0
    ) / 1_000_000.0

    # Add server-side timing header
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.4f}"

    return {
        "query": q,
        "k": k,
        "fuzzy": fuzzy,
        "latency_ms": round(elapsed_ms, 4),
        "count": len(results),
        "suggestions": results
    }


@app.post(
    "/api/insert",
    status_code=status.HTTP_201_CREATED
)
async def insert_word(payload: InsertRequest):
    """Dynamically inserts a word without rebuilding the index."""

    t0 = time.perf_counter_ns()

    try:
        is_new = engine.insert(
            word=payload.word,
            score=payload.score,
            script_override=payload.script
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )

    elapsed_ms = (
        time.perf_counter_ns() - t0
    ) / 1_000_000.0

    return {
        "word": payload.word,
        "is_new": is_new,
        "score": payload.score,
        "insert_latency_ms": round(elapsed_ms, 4),
        "status": "success"
    }


@app.post("/api/query")
async def record_query(payload: RecordQueryRequest):
    """Increments frequency score of query for dynamic reinforcement."""

    t0 = time.perf_counter_ns()

    try:
        new_freq = engine.record_query(
            query=payload.query,
            delta=payload.delta,
            script_override=payload.script
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )

    elapsed_ms = (
        time.perf_counter_ns() - t0
    ) / 1_000_000.0

    return {
        "query": payload.query,
        "updated_frequency": new_freq,
        "latency_ms": round(elapsed_ms, 4),
        "status": "success"
    }


@app.post("/api/delete")
async def delete_word(payload: DeleteRequest):
    """Tombstones a word from the engine."""

    try:
        deleted = engine.delete(
            payload.word,
            script_override=payload.script
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )

    return {
        "word": payload.word,
        "deleted": deleted,
        "status": "success" if deleted else "not_found"
    }


@app.get("/api/stats")
async def get_stats():
    """Returns engine memory, node counts, and compression metrics."""

    return engine.get_stats()


@app.get("/api/benchmark")
async def run_benchmark(
    size: int = Query(
        5000,
        ge=1000,
        le=100000,
        description="Benchmark dataset scale"
    )
):
    """Executes benchmark comparing Radix Tree against baselines."""

    runner = BenchmarkRunner(
        dataset_size=size
    )

    results = runner.run_all()

    return results