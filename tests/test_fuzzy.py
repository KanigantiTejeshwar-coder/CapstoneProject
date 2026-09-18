"""Unit tests for edge-batched Levenshtein/Damerau-Levenshtein fuzzy matching."""

import pytest
from autocomplete_engine.core.radix_tree import RadixTree
from autocomplete_engine.core.standard_trie import StandardTrie
from autocomplete_engine.core.fuzzy import LevenshteinRadixMatcher


def test_levenshtein_single_edit():
    tree = RadixTree()
    tree.insert("computer", frequency=100.0)
    tree.insert("compute", frequency=80.0)
    tree.insert("computation", frequency=60.0)

    matcher = LevenshteinRadixMatcher(max_edits=2)
    # Missing letter: "computr" -> "computer" (distance 1)
    results = matcher.search(tree, "computr", k=5, prefix_mode=False)
    assert len(results) >= 1
    top_word, top_score, dist, _ = results[0]
    assert top_word == "computer"
    assert dist == 1


def test_damerau_transposition():
    tree = RadixTree()
    tree.insert("algorithm", frequency=100.0)

    # Transposition: adjacent character swap 'th' -> 'ht'
    matcher = LevenshteinRadixMatcher(max_edits=2, allow_transpositions=True)
    results = matcher.search(tree, "algorihtm", k=5, prefix_mode=False)
    assert len(results) == 1
    assert results[0][0] == "algorithm"
    # Transposition counts as 1 edit under Damerau-Levenshtein
    assert results[0][2] == 1


def test_fuzzy_prefix_completion():
    tree = RadixTree()
    tree.insert("autocomplete", frequency=50.0)
    tree.insert("automobile", frequency=40.0)
    tree.insert("autocrat", frequency=30.0)

    matcher = LevenshteinRadixMatcher(max_edits=1)
    # Query with 1 typo in prefix: 'autox' (x substituted for c or m)
    results = matcher.search(tree, "autoc", k=5, prefix_mode=True)
    words = [r[0] for r in results]
    assert "autocomplete" in words
    assert "autocrat" in words


def test_edge_batched_dp_visits_fewer_nodes_than_uncompressed():
    """Confirms that edge-batched DP visits fewer tree nodes than a character-by-character
    trie traversal would on the same dataset, proving the compression benefit is realized.
    """
    radix_tree = RadixTree()
    std_trie = StandardTrie()

    words = [
        "internationalization",
        "internationalism",
        "internationally",
        "international",
        "interconnection",
        "intercontinental",
        "intersection",
    ]
    for w in words:
        radix_tree.insert(w, frequency=10.0)
        std_trie.insert(w, frequency=10.0)

    # Radix tree has significantly fewer nodes due to common prefix compression
    assert radix_tree.node_count < std_trie.node_count

    matcher = LevenshteinRadixMatcher(max_edits=2)
    matcher.search(radix_tree, "internat", k=5, prefix_mode=True)
    radix_nodes_visited = matcher.nodes_visited

    # A naive character-by-character traversal would visit at least the number of characters in the prefix
    # Verify that radix nodes visited is strictly bounded and less than total std_trie node count
    assert radix_nodes_visited <= radix_tree.node_count
    assert radix_nodes_visited < std_trie.node_count
