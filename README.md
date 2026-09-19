# Multilingual Autocomplete & Query Suggestion Engine

A high-performance query suggestion and autocomplete engine utilizing an **Edge-Compressed Radix Tree (Patricia Trie)** with Subtree Max-Weight Top-K Pruning, Unicode script routing, and Levenshtein typo correction.

---

## Key Features

1. **Edge-Compressed Radix Tree**:
   - Collapses non-branching character sequences into single nodes, drastically reducing node count and memory footprint compared to a standard trie.
   - Dynamic $O(|w|)$ in-place edge splitting and frequency updates without index rebuilds.
 <img width="1897" height="980" alt="image" src="https://github.com/user-attachments/assets/86082cf1-3714-4900-acc0-e381b708c7be" />
2. **Subtree Max-Weight Top-K Search**:
   - Maintains cached `max_subtree_weight` on nodes.
   - Best-first branch-and-bound search terminates early once the remaining subtrees cannot beat the current $k$-th candidate score ($O(|prefix| + K \log K)$).
   <img width="1901" height="978" alt="image" src="https://github.com/user-attachments/assets/234ba40f-4d31-47b5-9d03-fe47a334aafb" />

3. **Multilingual Support & Script Routing**:
   - Dedicated Radix Tree instances for **English**, **Telugu)**, **Tamil**, and **Hindi**.
   - Unicode NFKC normalization, casing, and accent-stripping routines that safely protect Indic vowel signs, viramas, and pulli marks.
   - Script-isolated routing prevents cross-script prefix pollution.
   <img width="1903" height="979" alt="image" src="https://github.com/user-attachments/assets/bf7e7b17-a97a-4f8b-8d18-45cce00fe588" />
<img width="1899" height="972" alt="image" src="https://github.com/user-attachments/assets/c1582f09-f665-46c7-9fe2-cdf6b303c8ce" />

4. **Typo Correction (Edge-Batched Levenshtein DP)**:
   - Dynamic programming row evaluation directly along compressed edges.
   - Early subtree branch pruning when the minimum row edit distance exceeds $d$.
   - Damerau-Levenshtein adjacent transposition support.
5. **Rigorous Head-to-Head Benchmarking**:
   - Evaluated against **Standard Trie** (with identical Top-K caching for fair comparison) and a **Hash Baseline**.
   - Tracks Memory (RSS & `tracemalloc`), in-process and HTTP query latency percentiles (P50, P90, P95, P99), dynamic insertion throughput, and typo suggestion accuracy (MRR and Precision@K per noise bucket).
   <img width="1698" height="262" alt="image" src="https://github.com/user-attachments/assets/67f07476-cce0-4d15-8346-51c334c38a7f" />
<img width="1706" height="970" alt="image" src="https://github.com/user-attachments/assets/4530a178-fed3-457a-8b81-e20143c81033" />

6. **REST API & Verification Demo**:
   - FastAPI server with timing headers (`X-Process-Time-Ms`).
   - Single-page vanilla HTML/JS demo (`web/index.html`).

---

## Directory Structure

```
capstone-1/
├── autocomplete_engine/
│   ├── api/
│   │   ├── __init__.py
│   │   └── server.py             # FastAPI REST endpoints
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── baselines.py          # Standard Trie & Hash Prefix Baselines
│   │   ├── benchmark_runner.py   # Full benchmark execution suite
│   │   └── datasets.py           # Multilingual Zipfian dataset & typo generator
│   ├── core/
│   │   ├── __init__.py
│   │   ├── engine.py             # Unified Autocomplete Engine coordinator
│   │   ├── fuzzy.py              # Edge-batched Levenshtein DP matcher
│   │   ├── node.py               # RadixNode and StandardTrieNode (__slots__)
│   │   ├── normalizer.py         # Unicode NFKC & script router
│   │   ├── persistence.py        # JSON and binary cold-start snapshotting
│   │   ├── radix_tree.py         # Compressed Trie with in-place edge split
│   │   └── standard_trie.py      # Baseline character-by-character trie
│   └── web/
│       └── index.html            # Verification UI
├── tests/
│   ├── test_benchmarks.py
│   ├── test_fairness.py          # Enforces fair baseline comparison contract
│   ├── test_fuzzy.py
│   ├── test_multilingual.py
│   └── test_radix_tree.py
├── future_work.md                # Documentation on edge re-merging & Pinyin
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Environment Setup
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\pip install -r requirements.txt
# On Linux/macOS:
.venv/bin/pip install -r requirements.txt
```

### 2. Run Tests
```bash
.venv\Scripts\pytest -v tests/
```

### 3. Run Benchmarks
```bash
.venv\Scripts\python -m autocomplete_engine.benchmarks.benchmark_runner --dataset-size 100000 --export results.json --export-md benchmark_report.md
```

### 4. Start REST API & Demo Server
```bash
.venv\Scripts\python -m uvicorn autocomplete_engine.api.server:app --reload --port 8000
```
Open `http://127.0.0.1:8000` in your web browser.

---

## API Endpoints

- `GET /api/suggest?q={prefix}&k={5}&fuzzy={true}&lang={latin}`: Returns Top-K completions and typo corrections with response latency header `X-Process-Time-Ms`.
- `POST /api/insert`: Dynamically inserts a word (`{"word": "...", "score": 100.0}`).
- `POST /api/query`: Dynamically reinforces query popularity (`{"query": "...", "delta": 1.0}`).
- `GET /api/stats`: Returns memory and compression statistics across all script trees.
- `GET /api/benchmark?size={5000}`: Triggers and returns comparative benchmark metrics.
