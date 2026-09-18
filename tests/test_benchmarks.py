"""Integration tests for BenchmarkRunner and API endpoints."""

import pytest
from fastapi.testclient import TestClient
from autocomplete_engine.api.server import app
from autocomplete_engine.benchmarks.benchmark_runner import BenchmarkRunner


def test_benchmark_runner_execution():
    """Confirms benchmark suite executes cleanly and reports all expected metric groups."""
    runner = BenchmarkRunner(dataset_size=1000)
    results = runner.run_all()

    assert "metadata" in results
    assert "memory_and_nodes" in results
    assert "query_latency_in_process" in results
    assert "accuracy" in results
    assert "dynamic_throughput" in results
    assert "cold_start" in results

    # Verify report generation
    report_md = runner.generate_markdown_report()
    assert "# Multilingual Autocomplete & Query Suggestion Engine: Benchmark Report" in report_md
    assert "Memory Footprint" in report_md


def test_api_endpoints():
    """Tests FastAPI endpoints with HTTP TestClient."""
    client = TestClient(app)

    # 1. Suggest exact & fuzzy
    res = client.get("/api/suggest?q=alg&k=3")
    assert res.status_code == 200
    data = res.json()
    assert "suggestions" in data
    assert "latency_ms" in data
    assert "X-Process-Time-Ms" in res.headers

    # 2. Dynamic insert
    ins_res = client.post("/api/insert", json={"word": "testdynamicword2026", "score": 999.0})
    assert ins_res.status_code == 201
    assert ins_res.json()["status"] == "success"

    # 3. Immediately query inserted word
    check_res = client.get("/api/suggest?q=testdynamicword&k=1")
    assert check_res.status_code == 200
    words = [s["word"] for s in check_res.json()["suggestions"]]
    assert "testdynamicword2026" in words

    # 4. Record query frequency increment
    rec_res = client.post("/api/query", json={"query": "testdynamicword2026", "delta": 10.0})
    assert rec_res.status_code == 200
    assert rec_res.json()["updated_frequency"] >= 1009.0

    # 5. Engine stats
    stats_res = client.get("/api/stats")
    assert stats_res.status_code == 200
    assert "total_words" in stats_res.json()
