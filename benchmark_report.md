# Multilingual Autocomplete & Query Suggestion Engine: Benchmark Report

**Dataset Size**: 100,000 entries | **Platform**: win32 | **Date**: 2026-09-10 15:25:35

> *Scope Note*: Evaluated strictly across English, Telugu, Tamil, and Hindi. Memory includes Python object overhead.

## 1. Memory Footprint & Structural Compression

| Metric | Radix Tree (Edge-Compressed) | Standard Trie (Baseline) | Hash / Sorted Baseline |
| :--- | :---: | :---: | :---: |
| Total Words | 100,000 | 100,000 | 100,000 |
| Total Nodes | **117,034** | 207,427 | 100,000 |
| Peak Memory | **21.29 MB** | 43.09 MB | 6.18 MB |
| Bytes per Word | **223.28 B** | 451.79 B | 64.8 B |
| Node Reduction vs Std Trie | **43.58%** | 0.0% | N/A |
| Build Time | 4.4094s | 3.3987s | 0.0559s |

## 2. In-Process Query Latency (Exact Prefix Top-K Retrieval)

| Architecture | Mean (µs) | P50 (µs) | P90 (µs) | P95 (µs) | P99 (µs) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Radix Tree (Top-K Pruned)** | **52.33** | **49.35** | **71.93** | **79.32** | **119.85** |
| Standard Trie (Top-K Pruned) | 57.86 | 58.8 | 68.41 | 72.69 | 132.92 |
| Hash Baseline (Bisect Scan) | 1597.21 | 257.05 | 485.14 | 622.58 | 33489.72 |

## 4. Dynamic Updates & Cold-Start Deserialization

- **Incremental Insert Throughput**: **131,853.06 ops/sec** (500 ops in 0.00379s)
- **Frequency Increment Throughput**: **176,211.45 ops/sec**
- **Cold-Start Deserialization (Binary Snapshot)**: **0.0285s** (5,000 entries)
- **Cold-Start Deserialization (JSON Snapshot)**: 0.0435s

## 5. Typo Correction Accuracy Partitioned by Noise Category

| Typo Category | Test Cases | Mean Reciprocal Rank (MRR) | Precision@5 | Hits@5 |
| :--- | :---: | :---: | :---: | :---: |
| `exact_prefix` | 50 | **0.1297** | **32.0%** | 16 / 50 |
| `1_edit` | 50 | **0.129** | **34.0%** | 17 / 50 |
| `2_edit` | 50 | **0.242** | **38.0%** | 19 / 50 |
| `transposition` | 50 | **0.1387** | **28.0%** | 14 / 50 |