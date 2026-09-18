# Future Work and Algorithmic Extensions

This document details future architectural optimizations and scope extensions for the Multilingual Autocomplete & Query Suggestion Engine.

---

## 1. Edge Re-Merging on Deletion

### Current Implementation:
The engine implements **minimal deletion via terminal tombstoning**:
- Setting `is_terminal = False` and resetting frequency to `0.0`.
- Propagating `recompute_max_weight()` up the ancestor path.
- The compressed edge hierarchy is kept intact.

### Full Edge Re-Merging Design:
To reclaim unused intermediate nodes and compress paths after deletion:
1. When a node's terminal flag is cleared and it has **exactly one child**:
   - Merge the child's edge label into the current node's incoming edge label.
   - Re-link parent to the merged child.
   - Decrement the overall node count.
2. If a parent node becomes non-terminal and has only one remaining child following a child deletion:
   - Perform a bottom-up edge concatenation.

---

## 2. Phonetic Romanized Transliteration for Indic Scripts (Telugu, Tamil, Hindi)

### Current Implementation:
- Queries in native Telugu, Tamil, and Hindi scripts match native prefixes with full Unicode NFKC normalization and diacritic safety.
- Cross-language prefix pollution is eliminated via isolated per-script Radix Trees.

### Proposed Extension:
- Support phonetic Romanized input (ITRANS / Harvard-Kyoto / WX transliteration scheme).
- Dual-path lookup: mapping English transliterated prefixes (e.g., `namas` -> `నమస్కారం` in Telugu and `नमस्ते` in Hindi; `vanak` -> `வணக்கம்` in Tamil).
- Phonetic n-gram language model candidate re-ranking.

---

## 3. High-Concurrency & Multi-Writer Synchronization

### Current Implementation:
- Single-writer lock (`threading.Lock` / `asyncio.Lock`) protecting writes (`insert`, `record_query`, `delete`).
- Reads (`search`) are non-blocking and execute without locks, trading momentary score freshness for maximal read throughput.

### Proposed Extension:
- Fine-grained per-node lock coupling (hand-over-hand locking) or Read-Copy-Update (RCU) / copy-on-write path pointers.
- Lock-free Radix Tree variants using atomic compare-and-swap (CAS) pointer updates.

---

## 4. Hardware Acceleration (SIMD / C++ / Rust Bindings)

- Porting the inner loop of `LevenshteinRadixMatcher` to Cython or C++ with AVX-512 / NEON vectorization.
- Bit-parallel Myers algorithm for edit distance calculations on short queries.
