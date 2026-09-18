"""Baseline Autocomplete implementations for comparative evaluation.

Includes:
1. StandardTrie (uncompressed character-by-character trie with identical Top-K cache).
2. HashPrefixBaseline (in-memory dict/list scanning for prefix matches).
"""

import bisect
from typing import List, Tuple, Dict, Any, Optional
from autocomplete_engine.core.standard_trie import StandardTrie


class HashPrefixBaseline:
    """Hash-based / Sorted List prefix baseline.
    
    Stores entries in a dictionary and maintains a sorted list of words
    for prefix range queries via binary search (bisect).
    """

    def __init__(self) -> None:
        self.dict_store: Dict[str, float] = {}
        self.sorted_words: List[str] = []
        self._is_sorted: bool = True

    def __len__(self) -> int:
        return len(self.dict_store)

    @property
    def node_count(self) -> int:
        """Hash table has 1 entry per word (no tree nodes)."""
        return len(self.dict_store)

    def insert(self, word: str, frequency: float = 1.0, data: Optional[Dict[str, Any]] = None) -> bool:
        """Inserts word into hash table and marks sorted list as needing re-sort."""
        is_new = word not in self.dict_store
        self.dict_store[word] = max(self.dict_store.get(word, 0.0), frequency)
        if is_new:
            self.sorted_words.append(word)
            self._is_sorted = False
        return is_new

    def _ensure_sorted(self) -> None:
        if not self._is_sorted:
            self.sorted_words.sort()
            self._is_sorted = True

    def top_k(self, prefix: str, k: int = 5) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        """Finds Top-K matching words starting with prefix."""
        if k <= 0:
            return []

        self._ensure_sorted()

        # Binary search for the start of the prefix range
        idx = bisect.bisect_left(self.sorted_words, prefix)
        matches: List[Tuple[str, float, Optional[Dict[str, Any]]]] = []

        while idx < len(self.sorted_words):
            word = self.sorted_words[idx]
            if not word.startswith(prefix):
                break
            matches.append((word, self.dict_store[word], None))
            idx += 1

        # Sort matches by frequency descending
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:k]

    def increment_frequency(self, word: str, delta: float = 1.0) -> float:
        """Increments word frequency dynamically."""
        if word in self.dict_store:
            self.dict_store[word] += delta
        else:
            self.insert(word, frequency=delta)
        return self.dict_store[word]

    def delete(self, word: str) -> bool:
        """Deletes word from hash store."""
        if word in self.dict_store:
            del self.dict_store[word]
            self.sorted_words = [w for w in self.sorted_words if w != word]
            return True
        return False
