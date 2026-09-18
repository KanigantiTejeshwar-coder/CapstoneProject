"""Fair-Comparison Contract Tests.

Asserts that StandardTrie and RadixTree expose the identical Top-K pruning
interface and caching mechanism (max_subtree_weight) to guarantee that
benchmarks isolate edge compression as the sole independent variable.
"""

import pytest
from autocomplete_engine.core.radix_tree import RadixTree
from autocomplete_engine.core.standard_trie import StandardTrie


def test_fair_caching_contract():
    radix = RadixTree()
    std = StandardTrie()

    # Verify both node types have max_subtree_weight in __slots__
    assert "max_subtree_weight" in radix.root.__slots__
    assert "max_subtree_weight" in std.root.__slots__

    dataset = [
        ("apple", 15.0),
        ("application", 40.0),
        ("apply", 25.0),
        ("banana", 10.0),
        ("band", 50.0),
        ("bandwidth", 35.0),
    ]

    for w, s in dataset:
        radix.insert(w, frequency=s)
        std.insert(w, frequency=s)

    # Root max_subtree_weight must match
    assert radix.root.max_subtree_weight == std.root.max_subtree_weight == 50.0

    # Prefix Top-K results and rankings must be strictly identical
    prefixes = ["ap", "app", "ban", "band", "b", ""]
    for p in prefixes:
        radix_res = [(w, score) for w, score, _ in radix.top_k(p, k=3)]
        std_res = [(w, score) for w, score, _ in std.top_k(p, k=3)]
        assert radix_res == std_res, f"Fairness violation for prefix '{p}': {radix_res} != {std_res}"


def test_dynamic_propagation_fairness():
    """Confirms both structures dynamically update max_subtree_weight upon frequency changes."""
    radix = RadixTree()
    std = StandardTrie()

    radix.insert("query", frequency=5.0)
    std.insert("query", frequency=5.0)

    # Increment
    radix.increment_frequency("query", delta=10.0)
    std.increment_frequency("query", delta=10.0)

    assert radix.root.max_subtree_weight == std.root.max_subtree_weight == 15.0
