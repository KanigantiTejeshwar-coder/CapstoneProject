"""Comprehensive Benchmarking Suite for Multilingual Autocomplete Engine.

Measures:
1. Memory Footprint (total bytes, bytes/word, node count reduction ratio vs Standard Trie).
2. Query Latency (P50, P90, P95, P99, mean) for RadixTree, StandardTrie, and HashBaseline.
3. Cold-Start Deserialization Latency (Binary vs JSON snapshot).
4. Dynamic Insert Throughput (operations/sec under single-writer lock).
5. Typo Suggestion Accuracy (MRR and Precision@K partitioned by typo bucket).
"""

import argparse
import gc
import json
import os
import sys
import tempfile
import time
import tracemalloc
from typing import Dict, Any, List, Tuple
import numpy as np

from autocomplete_engine.core.radix_tree import RadixTree
from autocomplete_engine.core.standard_trie import StandardTrie
from autocomplete_engine.core.engine import AutocompleteEngine
from autocomplete_engine.core.persistence import TreePersistence
from autocomplete_engine.benchmarks.baselines import HashPrefixBaseline
from autocomplete_engine.benchmarks.datasets import MultilingualDatasetGenerator


class BenchmarkRunner:
    """Executes head-to-head benchmarks between Radix Tree and baseline systems."""

    def __init__(self, dataset_size: int = 10000, seed: int = 42) -> None:
        self.dataset_size = dataset_size
        self.generator = MultilingualDatasetGenerator(seed=seed)
        self.results: Dict[str, Any] = {}

    def run_all(self, test_client=None) -> Dict[str, Any]:
        """Runs the entire benchmark suite and aggregates results."""
        print(f"[*] Generating multilingual dataset of {self.dataset_size:,} words...")
        vocab = self.generator.generate_vocabulary(total_words=self.dataset_size)

        print("[*] Generating labeled typo benchmark queries...")
        typo_buckets = self.generator.generate_typo_benchmark_set(vocab, samples_per_bucket=50)

        # 1. Memory & Construction Benchmark
        print("[*] Running Memory & Construction Benchmarks...")
        mem_results = self._benchmark_memory_and_construction(vocab)

        # 2. Dynamic Update Throughput Benchmark
        print("[*] Running Dynamic Update Throughput Benchmark...")
        throughput_results = self._benchmark_dynamic_throughput(vocab[:1000])

        # 3. Persistence / Cold-Start Benchmark
        print("[*] Running Cold-Start Persistence Benchmark...")
        cold_start_results = self._benchmark_cold_start(vocab)

        # 4. In-Process Query Latency Benchmark
        print("[*] Running In-Process Query Latency Benchmarks...")
        latency_results = self._benchmark_query_latency(vocab, typo_buckets)

        # 5. Accuracy Benchmark (MRR & Precision@K)
        print("[*] Running Typo Suggestion Accuracy Benchmarks...")
        accuracy_results = self._benchmark_accuracy(vocab, typo_buckets)

        # 6. HTTP Round-Trip Latency (if client provided)
        http_latency_results = None
        if test_client is not None:
            print("[*] Running HTTP Round-Trip Latency Benchmark...")
            http_latency_results = self._benchmark_http_latency(test_client, typo_buckets)

        self.results = {
            "metadata": {
                "dataset_size": self.dataset_size,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "python_version": sys.version.split()[0],
                "platform": sys.platform,
                "scope_note": "Evaluated strictly across English, Telugu, Tamil, and Hindi. Memory includes Python object overhead."
            },
            "memory_and_nodes": mem_results,
            "dynamic_throughput": throughput_results,
            "cold_start": cold_start_results,
            "query_latency_in_process": latency_results,
            "http_roundtrip_latency": http_latency_results,
            "accuracy": accuracy_results,
        }
        return self.results

    def _benchmark_memory_and_construction(self, vocab: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Compares memory usage and node count across structures."""
        results = {}

        # Radix Tree
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        radix_tree = RadixTree()
        for w, s in vocab:
            radix_tree.insert(w, frequency=s)
        t_radix = time.perf_counter() - t0
        _, peak_radix = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results["radix_tree"] = {
            "total_nodes": radix_tree.node_count,
            "total_words": len(radix_tree),
            "peak_memory_bytes": peak_radix,
            "bytes_per_word": round(peak_radix / len(vocab), 2),
            "build_time_seconds": round(t_radix, 4)
        }

        # Standard Trie
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        std_trie = StandardTrie()
        for w, s in vocab:
            std_trie.insert(w, frequency=s)
        t_std = time.perf_counter() - t0
        _, peak_std = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results["standard_trie"] = {
            "total_nodes": std_trie.node_count,
            "total_words": len(std_trie),
            "peak_memory_bytes": peak_std,
            "bytes_per_word": round(peak_std / len(vocab), 2),
            "build_time_seconds": round(t_std, 4)
        }

        # Hash Baseline
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        hash_base = HashPrefixBaseline()
        for w, s in vocab:
            hash_base.insert(w, frequency=s)
        t_hash = time.perf_counter() - t0
        _, peak_hash = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results["hash_baseline"] = {
            "total_nodes": hash_base.node_count,
            "total_words": len(hash_base),
            "peak_memory_bytes": peak_hash,
            "bytes_per_word": round(peak_hash / len(vocab), 2),
            "build_time_seconds": round(t_hash, 4)
        }

        # Comparisons
        node_reduction = (
            (1.0 - (radix_tree.node_count / std_trie.node_count)) * 100
            if std_trie.node_count > 0 else 0.0
        )
        mem_reduction = (
            (1.0 - (peak_radix / peak_std)) * 100
            if peak_std > 0 else 0.0
        )

        results["comparison"] = {
            "node_reduction_pct_vs_std_trie": round(node_reduction, 2),
            "memory_reduction_pct_vs_std_trie": round(mem_reduction, 2),
        }

        return results

    def _benchmark_dynamic_throughput(self, sample_words: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Measures dynamic insertion throughput (ops/sec) without full rebuild."""
        radix = RadixTree()
        # Pre-seed with half
        half = len(sample_words) // 2
        for w, s in sample_words[:half]:
            radix.insert(w, frequency=s)

        to_insert = sample_words[half:]
        t0 = time.perf_counter()
        for w, s in to_insert:
            radix.insert(w, frequency=s)
        elapsed = time.perf_counter() - t0
        ops_per_sec = len(to_insert) / elapsed if elapsed > 0 else 0.0

        # Dynamic frequency increments
        t0 = time.perf_counter()
        for w, _ in to_insert:
            radix.increment_frequency(w, delta=5.0)
        elapsed_inc = time.perf_counter() - t0
        inc_ops_per_sec = len(to_insert) / elapsed_inc if elapsed_inc > 0 else 0.0

        return {
            "insert_operations": len(to_insert),
            "insert_elapsed_seconds": round(elapsed, 5),
            "insert_ops_per_sec": round(ops_per_sec, 2),
            "increment_elapsed_seconds": round(elapsed_inc, 5),
            "increment_ops_per_sec": round(inc_ops_per_sec, 2),
        }

    def _benchmark_cold_start(self, vocab: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Measures serialization and deserialization time (cold-start latency)."""
        radix = RadixTree()
        # Take 5,000 items to test snapshotting efficiently
        subset = vocab[:min(5000, len(vocab))]
        for w, s in subset:
            radix.insert(w, frequency=s)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "snapshot.json")
            bin_path = os.path.join(tmpdir, "snapshot.bin")

            save_json_time = TreePersistence.save_json(radix, json_path)
            _, load_json_time = TreePersistence.load_json(json_path)

            save_bin_time = TreePersistence.save_binary(radix, bin_path)
            _, load_bin_time = TreePersistence.load_binary(bin_path)

        return {
            "tested_entries": len(subset),
            "json_save_seconds": round(save_json_time, 4),
            "json_load_seconds": round(load_json_time, 4),
            "binary_save_seconds": round(save_bin_time, 4),
            "binary_load_seconds": round(load_bin_time, 4),
        }

    def _benchmark_query_latency(
        self,
        vocab: List[Tuple[str, float]],
        typo_buckets: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Measures P50, P90, P95, P99, and mean latency in microseconds for in-process calls."""
        # Build populated trees
        radix = RadixTree()
        std_trie = StandardTrie()
        hash_base = HashPrefixBaseline()

        for w, s in vocab:
            radix.insert(w, frequency=s)
            std_trie.insert(w, frequency=s)
            hash_base.insert(w, frequency=s)

        exact_queries = [tc["query"] for tc in typo_buckets["exact_prefix"]]
        if not exact_queries:
            exact_queries = [w[:3] for w, _ in vocab[:50]]

        # Latency collector helper
        def measure_latencies(fn, queries):
            lats = []
            for q in queries:
                t0 = time.perf_counter_ns()
                _ = fn(q)
                lats.append((time.perf_counter_ns() - t0) / 1000.0)  # Microseconds
            return {
                "mean_us": round(float(np.mean(lats)), 2),
                "p50_us": round(float(np.percentile(lats, 50)), 2),
                "p90_us": round(float(np.percentile(lats, 90)), 2),
                "p95_us": round(float(np.percentile(lats, 95)), 2),
                "p99_us": round(float(np.percentile(lats, 99)), 2),
            }

        radix_lat = measure_latencies(lambda q: radix.top_k(q, k=5), exact_queries)
        std_lat = measure_latencies(lambda q: std_trie.top_k(q, k=5), exact_queries)
        hash_lat = measure_latencies(lambda q: hash_base.top_k(q, k=5), exact_queries)

        return {
            "query_count": len(exact_queries),
            "radix_tree": radix_lat,
            "standard_trie": std_lat,
            "hash_baseline": hash_lat,
        }

    def _benchmark_http_latency(
        self,
        test_client,
        typo_buckets: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Measures HTTP round-trip latency through FastAPI test client."""
        queries = [tc["query"] for tc in typo_buckets["exact_prefix"]][:30]
        lats = []
        for q in queries:
            t0 = time.perf_counter_ns()
            resp = test_client.get(f"/api/suggest?q={q}&k=5")
            lats.append((time.perf_counter_ns() - t0) / 1000.0)  # Microseconds

        return {
            "requests_count": len(queries),
            "mean_us": round(float(np.mean(lats)), 2),
            "p50_us": round(float(np.percentile(lats, 50)), 2),
            "p90_us": round(float(np.percentile(lats, 90)), 2),
            "p95_us": round(float(np.percentile(lats, 95)), 2),
            "p99_us": round(float(np.percentile(lats, 99)), 2),
        }

    def _benchmark_accuracy(
        self,
        vocab: List[Tuple[str, float]],
        typo_buckets: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Evaluates MRR and Precision@5 separately for each typo bucket."""
        engine = AutocompleteEngine()
        for w, s in vocab:
            engine.insert(w, score=s)

        bucket_results = {}

        for bucket_name, test_cases in typo_buckets.items():
            reciprocal_ranks = []
            hits_at_k = 0
            k = 5

            for tc in test_cases:
                query = tc["query"]
                target = tc["target"]

                # Enable fuzzy for typo buckets, allow exact for exact_prefix
                suggestions = engine.search(query, k=k, fuzzy=True)
                suggested_words = [s["word"] for s in suggestions]

                if target in suggested_words:
                    rank = suggested_words.index(target) + 1
                    reciprocal_ranks.append(1.0 / rank)
                    hits_at_k += 1
                else:
                    reciprocal_ranks.append(0.0)

            mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
            precision_at_k = (hits_at_k / len(test_cases)) if test_cases else 0.0

            bucket_results[bucket_name] = {
                "test_cases": len(test_cases),
                "mrr": round(mrr, 4),
                "precision_at_5": round(precision_at_k, 4),
                "hits_at_5": hits_at_k,
            }

        return bucket_results

    def generate_markdown_report(self) -> str:
        """Generates a comprehensive GitHub-flavored Markdown table report."""
        r = self.results
        mem = r.get("memory_and_nodes", {})
        radix_mem = mem.get("radix_tree", {})
        std_mem = mem.get("standard_trie", {})
        hash_mem = mem.get("hash_baseline", {})
        comp = mem.get("comparison", {})
        lat = r.get("query_latency_in_process", {})
        acc = r.get("accuracy", {})
        tp = r.get("dynamic_throughput", {})
        cs = r.get("cold_start", {})
        http_lat = r.get("http_roundtrip_latency")

        md = []
        md.append("# Multilingual Autocomplete & Query Suggestion Engine: Benchmark Report\n")
        md.append(f"**Dataset Size**: {r['metadata']['dataset_size']:,} entries | **Platform**: {r['metadata']['platform']} | **Date**: {r['metadata']['timestamp']}\n")
        md.append(f"> *Scope Note*: {r['metadata']['scope_note']}\n")

        md.append("## 1. Memory Footprint & Structural Compression\n")
        md.append("| Metric | Radix Tree (Edge-Compressed) | Standard Trie (Baseline) | Hash / Sorted Baseline |")
        md.append("| :--- | :---: | :---: | :---: |")
        md.append(f"| Total Words | {radix_mem.get('total_words', 0):,} | {std_mem.get('total_words', 0):,} | {hash_mem.get('total_words', 0):,} |")
        md.append(f"| Total Nodes | **{radix_mem.get('total_nodes', 0):,}** | {std_mem.get('total_nodes', 0):,} | {hash_mem.get('total_nodes', 0):,} |")
        md.append(f"| Peak Memory | **{radix_mem.get('peak_memory_bytes', 0)/(1024*1024):.2f} MB** | {std_mem.get('peak_memory_bytes', 0)/(1024*1024):.2f} MB | {hash_mem.get('peak_memory_bytes', 0)/(1024*1024):.2f} MB |")
        md.append(f"| Bytes per Word | **{radix_mem.get('bytes_per_word', 0)} B** | {std_mem.get('bytes_per_word', 0)} B | {hash_mem.get('bytes_per_word', 0)} B |")
        md.append(f"| Node Reduction vs Std Trie | **{comp.get('node_reduction_pct_vs_std_trie', 0)}%** | 0.0% | N/A |")
        md.append(f"| Build Time | {radix_mem.get('build_time_seconds', 0)}s | {std_mem.get('build_time_seconds', 0)}s | {hash_mem.get('build_time_seconds', 0)}s |\n")

        md.append("## 2. In-Process Query Latency (Exact Prefix Top-K Retrieval)\n")
        md.append("| Architecture | Mean (µs) | P50 (µs) | P90 (µs) | P95 (µs) | P99 (µs) |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        r_l = lat.get("radix_tree", {})
        s_l = lat.get("standard_trie", {})
        h_l = lat.get("hash_baseline", {})
        md.append(f"| **Radix Tree (Top-K Pruned)** | **{r_l.get('mean_us', 0)}** | **{r_l.get('p50_us', 0)}** | **{r_l.get('p90_us', 0)}** | **{r_l.get('p95_us', 0)}** | **{r_l.get('p99_us', 0)}** |")
        md.append(f"| Standard Trie (Top-K Pruned) | {s_l.get('mean_us', 0)} | {s_l.get('p50_us', 0)} | {s_l.get('p90_us', 0)} | {s_l.get('p95_us', 0)} | {s_l.get('p99_us', 0)} |")
        md.append(f"| Hash Baseline (Bisect Scan) | {h_l.get('mean_us', 0)} | {h_l.get('p50_us', 0)} | {h_l.get('p90_us', 0)} | {h_l.get('p95_us', 0)} | {h_l.get('p99_us', 0)} |\n")

        if http_lat:
            md.append("## 3. Full HTTP Round-Trip Latency (FastAPI Endpoint)\n")
            md.append("| Protocol | Requests | Mean (µs) | P50 (µs) | P95 (µs) | P99 (µs) |")
            md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
            md.append(f"| HTTP GET `/api/suggest` | {http_lat.get('requests_count', 0)} | {http_lat.get('mean_us', 0)} | {http_lat.get('p50_us', 0)} | {http_lat.get('p95_us', 0)} | {http_lat.get('p99_us', 0)} |\n")

        md.append("## 4. Dynamic Updates & Cold-Start Deserialization\n")
        md.append(f"- **Incremental Insert Throughput**: **{tp.get('insert_ops_per_sec', 0):,} ops/sec** ({tp.get('insert_operations', 0)} ops in {tp.get('insert_elapsed_seconds', 0)}s)")
        md.append(f"- **Frequency Increment Throughput**: **{tp.get('increment_ops_per_sec', 0):,} ops/sec**")
        md.append(f"- **Cold-Start Deserialization (Binary Snapshot)**: **{cs.get('binary_load_seconds', 0)}s** ({cs.get('tested_entries', 0):,} entries)")
        md.append(f"- **Cold-Start Deserialization (JSON Snapshot)**: {cs.get('json_load_seconds', 0)}s\n")

        md.append("## 5. Typo Correction Accuracy Partitioned by Noise Category\n")
        md.append("| Typo Category | Test Cases | Mean Reciprocal Rank (MRR) | Precision@5 | Hits@5 |")
        md.append("| :--- | :---: | :---: | :---: | :---: |")
        for bucket, metrics in acc.items():
            md.append(f"| `{bucket}` | {metrics.get('test_cases', 0)} | **{metrics.get('mrr', 0)}** | **{metrics.get('precision_at_5', 0)*100:.1f}%** | {metrics.get('hits_at_5', 0)} / {metrics.get('test_cases', 0)} |")

        return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Run Autocomplete Benchmarks")
    parser.add_argument("--dataset-size", type=int, default=10000, help="Vocabulary size to benchmark")
    parser.add_argument("--export", type=str, default="", help="JSON export filepath")
    parser.add_argument("--export-md", type=str, default="", help="Markdown report export filepath")
    args = parser.parse_args()

    runner = BenchmarkRunner(dataset_size=args.dataset_size)
    runner.run_all()

    report_md = runner.generate_markdown_report()
    print("\n" + report_md)

    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(runner.results, f, indent=2)
        print(f"\n[+] Exported JSON results to {args.export}")

    if args.export_md:
        with open(args.export_md, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"[+] Exported Markdown report to {args.export_md}")


if __name__ == "__main__":
    main()
