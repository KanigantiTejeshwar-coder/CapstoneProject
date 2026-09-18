"""Standard uncompressed character-by-character Trie.

Implements the identical Top-K pruning and `max_subtree_weight` propagation as
`RadixTree` to isolate edge-compression as the sole independent variable
in benchmarks.
"""

import heapq
from typing import List, Tuple, Optional, Dict, Any
from autocomplete_engine.core.node import StandardTrieNode


class StandardTrie:
    """Standard character-by-character Trie implementation."""

    def __init__(self) -> None:
        self.root: StandardTrieNode = StandardTrieNode(char="", is_terminal=False, frequency=0.0)
        self._size: int = 0
        self._node_count: int = 1

    def __len__(self) -> int:
        return self._size

    @property
    def node_count(self) -> int:
        return self._node_count

    def insert(
        self,
        word: str,
        frequency: float = 1.0,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Inserts a word character by character, propagating max_subtree_weight."""
        ancestor_path: List[StandardTrieNode] = [self.root]
        curr = self.root

        for ch in word:
            if ch not in curr.children:
                child = StandardTrieNode(char=ch, is_terminal=False, frequency=0.0)
                curr.children[ch] = child
                self._node_count += 1
            curr = curr.children[ch]
            ancestor_path.append(curr)

        is_new_word = not curr.is_terminal
        curr.is_terminal = True
        curr.frequency = max(curr.frequency, frequency)
        if data is not None:
            curr.data = data
        if is_new_word:
            self._size += 1

        # Identical upward propagation of max_subtree_weight
        for node in reversed(ancestor_path):
            node.recompute_max_weight()

        return is_new_word

    def increment_frequency(self, word: str, delta: float = 1.0) -> float:
        """Increments word frequency dynamically."""
        ancestor_path: List[StandardTrieNode] = [self.root]
        curr = self.root

        for ch in word:
            if ch not in curr.children:
                child = StandardTrieNode(char=ch, is_terminal=False, frequency=0.0)
                curr.children[ch] = child
                self._node_count += 1
            curr = curr.children[ch]
            ancestor_path.append(curr)

        if not curr.is_terminal:
            curr.is_terminal = True
            curr.frequency = delta
            self._size += 1
        else:
            curr.frequency += delta

        for node in reversed(ancestor_path):
            node.recompute_max_weight()

        return curr.frequency

    def delete(self, word: str) -> bool:
        """Tombstones the terminal flag of a word (identical to RadixTree deletion)."""
        ancestor_path: List[StandardTrieNode] = [self.root]
        curr = self.root

        for ch in word:
            if ch not in curr.children:
                return False
            curr = curr.children[ch]
            ancestor_path.append(curr)

        if curr.is_terminal:
            curr.is_terminal = False
            curr.frequency = 0.0
            self._size -= 1
            for node in reversed(ancestor_path):
                node.recompute_max_weight()
            return True

        return False

    def contains(self, word: str) -> bool:
        """Checks if exact word exists."""
        curr = self.root
        for ch in word:
            if ch not in curr.children:
                return False
            curr = curr.children[ch]
        return curr.is_terminal

    def get_frequency(self, word: str) -> Optional[float]:
        """Gets frequency score of word if present."""
        curr = self.root
        for ch in word:
            if ch not in curr.children:
                return None
            curr = curr.children[ch]
        return curr.frequency if curr.is_terminal else None

    def search_prefix_node(self, prefix: str) -> Optional[Tuple[StandardTrieNode, str]]:
        """Walks to prefix node."""
        curr = self.root
        for ch in prefix:
            if ch not in curr.children:
                return None
            curr = curr.children[ch]
        return curr, prefix

    def top_k(self, prefix: str, k: int = 5) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        """Finds Top-K highest frequency completions using the identical branch-and-bound algorithm."""
        if k <= 0:
            return []

        match_res = self.search_prefix_node(prefix)
        if match_res is None:
            return []

        start_node, matched_path = match_res

        heap: List[Tuple[float, str, StandardTrieNode]] = []
        heapq.heappush(heap, (-start_node.max_subtree_weight, matched_path, start_node))

        results: List[Tuple[str, float, Optional[Dict[str, Any]]]] = []

        while heap:
            neg_max_w, word_acc, node = heapq.heappop(heap)
            max_w = -neg_max_w

            if len(results) >= k and max_w <= results[-1][1]:
                break

            if node.is_terminal:
                results.append((word_acc, node.frequency, node.data))
                results.sort(key=lambda item: item[1], reverse=True)
                if len(results) > k:
                    results.pop()

            for child in node.children.values():
                if len(results) >= k and child.max_subtree_weight <= results[-1][1]:
                    continue
                heapq.heappush(
                    heap,
                    (-child.max_subtree_weight, word_acc + child.char, child)
                )

        return results[:k]
