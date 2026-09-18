"""Unified Autocomplete Engine coordinating multilingual script routing,
concurrency control, fuzzy typo correction, and Top-K retrieval.
"""

import threading
from typing import List, Dict, Any, Optional, Tuple

from autocomplete_engine.core.radix_tree import RadixTree
from autocomplete_engine.core.normalizer import UnicodeNormalizer, ScriptType
from autocomplete_engine.core.fuzzy import LevenshteinRadixMatcher


class AutocompleteEngine:
    """Multilingual Query Suggestion & Autocomplete Engine.

    Concurrency Model:
        - Single-writer model: engine uses a threading.Lock around write operations
          (`insert`, `record_query`, `delete`, `batch_load`).
        - Lock-free reads: `search` reads without acquiring the write lock.
          Reads may observe a slightly stale `max_subtree_weight` mid-update,
          which is acceptable for autocomplete suggestions.
        - "Dynamic updates without full rebuild": new words and frequency increments
          modify the tree in-place in O(|w|) time without rebuilding indices.

    Script Routing:
        - Maintains dedicated RadixTree instances per Unicode script
          (Latin, Telugu, Tamil, Devanagari, Other).
    """

    def __init__(
        self,
        strip_accents: bool = False,
        lowercase: bool = True,
        max_edits: int = 2
    ) -> None:
        self.normalizer = UnicodeNormalizer(
            strip_accents=strip_accents,
            lowercase=lowercase
        )

        self.fuzzy_matcher = LevenshteinRadixMatcher(
            max_edits=max_edits,
            allow_transpositions=True
        )

        self._write_lock = threading.Lock()

        # Dedicated trie per supported script.
        self.trees: Dict[ScriptType, RadixTree] = {
            ScriptType.LATIN: RadixTree(),
            ScriptType.TELUGU: RadixTree(),
            ScriptType.TAMIL: RadixTree(),
            ScriptType.DEVANAGARI: RadixTree(),
            ScriptType.OTHER: RadixTree(),
        }

    def _route(
        self,
        text: str,
        script_override: Optional[str] = None
    ) -> Tuple[RadixTree, ScriptType]:
        """Routes text to the appropriate script trie."""

        if script_override:
            try:
                st = ScriptType(script_override.lower())
                return self.trees[st], st
            except ValueError:
                raise ValueError(
                    f"Unsupported script_override: {script_override}"
                )

        st = self.normalizer.detect_script(text)
        return self.trees[st], st

    def insert(
        self,
        word: str,
        score: float = 1.0,
        data: Optional[Dict[str, Any]] = None,
        script_override: Optional[str] = None
    ) -> bool:
        """Inserts a word into the engine under the single-writer lock.

        Returns:
            True if newly inserted, False if already present and updated.
        """
        norm_word = self.normalizer.normalize(word)

        if not norm_word:
            return False

        tree, _ = self._route(norm_word, script_override)

        with self._write_lock:
            return tree.insert(
                norm_word,
                frequency=score,
                data=data
            )

    def record_query(
        self,
        query: str,
        delta: float = 1.0,
        script_override: Optional[str] = None
    ) -> float:
        """Dynamically reinforces query frequency under the single-writer lock."""
        norm_query = self.normalizer.normalize(query)

        if not norm_query:
            return 0.0

        tree, _ = self._route(norm_query, script_override)

        with self._write_lock:
            return tree.increment_frequency(
                norm_query,
                delta=delta
            )

    def delete(
        self,
        word: str,
        script_override: Optional[str] = None
    ) -> bool:
        """Tombstones a word under the single-writer lock."""
        norm_word = self.normalizer.normalize(word)

        if not norm_word:
            return False

        tree, _ = self._route(norm_word, script_override)

        with self._write_lock:
            return tree.delete(norm_word)

    def batch_load(
        self,
        items: List[Tuple[str, float]],
        script_override: Optional[str] = None
    ) -> int:
        """Batch inserts words under a single write lock acquisition."""
        inserted_count = 0

        with self._write_lock:
            for word, score in items:
                norm_word = self.normalizer.normalize(word)

                if norm_word:
                    tree, _ = self._route(
                        norm_word,
                        script_override
                    )

                    if tree.insert(
                        norm_word,
                        frequency=score
                    ):
                        inserted_count += 1

        return inserted_count

    def search(
        self,
        prefix: str,
        k: int = 5,
        fuzzy: bool = True,
        script_override: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Searches for Top-K query suggestions with optional typo correction.

        Lock-free read: multiple callers search concurrently without blocking.

        Returns:
            List of dicts:
            [{"word": str, "score": float, "edit_distance": int,
              "match_type": str, "data": dict}]
        """

        norm_prefix = self.normalizer.normalize(prefix)

        if not norm_prefix:
            # Empty prefix: return global top-k from default/latin tree.
            tree, st = self._route("", script_override)

            raw = tree.top_k("", k=k)

            return [
                {
                    "word": w,
                    "score": s,
                    "edit_distance": 0,
                    "match_type": "exact",
                    "script": st.value,
                    "data": d
                }
                for w, s, d in raw
            ]

        tree, script_type = self._route(
            norm_prefix,
            script_override
        )

        # 1. Exact prefix search via Branch-and-Bound Top-K.
        exact_results = tree.top_k(
            norm_prefix,
            k=k
        )

        suggestions: List[Dict[str, Any]] = [
            {
                "word": w,
                "score": s,
                "edit_distance": 0,
                "match_type": "exact",
                "script": script_type.value,
                "data": d
            }
            for w, s, d in exact_results
        ]

        # 2. If we need more candidates and fuzzy search is enabled.
        if fuzzy and len(suggestions) < k:
            remaining_k = k - len(suggestions)

            seen_words = {
                s["word"]
                for s in suggestions
            }

            # Adjust max edits based on query length
            # without modifying the shared fuzzy matcher.
            effective_max_edits = min(
                self.fuzzy_matcher.max_edits,
                1 if len(norm_prefix) <= 3 else 2
            )

            # Create a request-local matcher so concurrent searches
            # cannot modify each other's fuzzy-search configuration.
            matcher = LevenshteinRadixMatcher(
                max_edits=effective_max_edits,
                allow_transpositions=True
            )

            fuzzy_matches = matcher.search(
                tree=tree,
                query=norm_prefix,
                k=remaining_k,
                prefix_mode=True
            )

            for w, score, dist, d in fuzzy_matches:
                if w not in seen_words and dist > 0:
                    seen_words.add(w)

                    suggestions.append(
                        {
                            "word": w,
                            "score": score,
                            "edit_distance": dist,
                            "match_type": "fuzzy",
                            "script": script_type.value,
                            "data": d
                        }
                    )

                    if len(suggestions) >= k:
                        break

        return suggestions[:k]

    def get_stats(self) -> Dict[str, Any]:
        """Returns engine statistics across all script trees."""

        total_words = 0
        total_nodes = 0
        script_stats = {}

        for st, tree in self.trees.items():
            w_cnt = len(tree)
            n_cnt = tree.node_count

            total_words += w_cnt
            total_nodes += n_cnt

            script_stats[st.value] = {
                "words": w_cnt,
                "nodes": n_cnt,
                "compression_ratio": (
                    n_cnt / w_cnt
                    if w_cnt > 0
                    else 1.0
                )
            }

        return {
            "total_words": total_words,
            "total_nodes": total_nodes,
            "overall_compression_ratio": (
                total_nodes / total_words
                if total_words > 0
                else 1.0
            ),
            "scripts": script_stats,
        }