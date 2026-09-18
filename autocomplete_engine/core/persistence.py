"""Serialization and deserialization for RadixTree and StandardTrie.

Supports JSON and high-speed binary snapshotting. Deserialization time
for cold-start is benchmarked as a distinct metric.
"""

import json
import pickle
import time
from typing import Tuple, Any, Union
from autocomplete_engine.core.radix_tree import RadixTree
from autocomplete_engine.core.standard_trie import StandardTrie
from autocomplete_engine.core.node import RadixNode, StandardTrieNode


class TreePersistence:
    """Handles snapshotting and cold-start loading for trie structures."""

    @staticmethod
    def save_json(tree: Union[RadixTree, StandardTrie], filepath: str) -> float:
        """Saves tree to JSON. Returns elapsed write time in seconds."""
        t0 = time.perf_counter()
        is_radix = isinstance(tree, RadixTree)
        data = {
            "type": "radix" if is_radix else "standard",
            "size": len(tree),
            "node_count": tree.node_count,
            "root": tree.root.to_dict(),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return time.perf_counter() - t0

    @staticmethod
    def load_json(filepath: str) -> Tuple[Union[RadixTree, StandardTrie], float]:
        """Loads tree from JSON. Returns (tree, elapsed_seconds)."""
        t0 = time.perf_counter()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        is_radix = data.get("type") == "radix"
        if is_radix:
            tree = RadixTree()
            tree.root = RadixNode.from_dict(data["root"])
            tree._size = data.get("size", 0)
            tree._node_count = data.get("node_count", 0)
        else:
            tree = StandardTrie()
            tree.root = StandardTrieNode.from_dict(data["root"])
            tree._size = data.get("size", 0)
            tree._node_count = data.get("node_count", 0)

        elapsed = time.perf_counter() - t0
        return tree, elapsed

    @staticmethod
    def save_binary(tree: Union[RadixTree, StandardTrie], filepath: str) -> float:
        """Saves tree to binary snapshot using pickle protocol 5. Returns write time in seconds."""
        t0 = time.perf_counter()
        with open(filepath, "wb") as f:
            pickle.dump(tree, f, protocol=pickle.HIGHEST_PROTOCOL)
        return time.perf_counter() - t0

    @staticmethod
    def load_binary(filepath: str) -> Tuple[Union[RadixTree, StandardTrie], float]:
        """Loads tree from binary snapshot. Returns (tree, elapsed_seconds)."""
        t0 = time.perf_counter()
        with open(filepath, "rb") as f:
            tree = pickle.load(f)
        elapsed = time.perf_counter() - t0
        return tree, elapsed
