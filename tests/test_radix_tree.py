"""Unit tests for RadixTree data structure."""

import pytest
from autocomplete_engine.core.radix_tree import RadixTree


def test_insert_and_contains():
    tree = RadixTree()
    words = ["romane", "romanus", "romulus", "rubens", "ruber", "rubicon", "rubicundus"]
    for w in words:
        assert tree.insert(w, frequency=float(len(w)))

    assert len(tree) == len(words)
    # Check all inserted words are present
    for w in words:
        assert tree.contains(w)
        assert tree.get_frequency(w) == float(len(w))

    # Check non-inserted words
    assert not tree.contains("roma")
    assert not tree.contains("rubic")
    assert not tree.contains("unknown")


def test_edge_splitting():
    tree = RadixTree()
    # Inserting 'test' creates 1 leaf
    tree.insert("test", frequency=10.0)
    assert tree.node_count == 2  # root + 'test'

    # Inserting 'team' splits 'test' at 'te' -> intermediate 'te', children 'st', 'am'
    tree.insert("team", frequency=20.0)
    assert tree.node_count == 4  # root, 'te', 'st', 'am'
    assert tree.contains("test")
    assert tree.contains("team")

    # Inserting 'tea' ends exactly at intermediate node 'tea' after splitting 'team'
    tree.insert("tea", frequency=30.0)
    assert tree.contains("tea")
    assert tree.contains("team")
    assert tree.contains("test")


def test_dynamic_frequency_increment():
    tree = RadixTree()
    tree.insert("apple", frequency=5.0)
    assert tree.get_frequency("apple") == 5.0

    # Increment existing
    new_f = tree.increment_frequency("apple", delta=2.5)
    assert new_f == 7.5
    assert tree.get_frequency("apple") == 7.5
    assert tree.root.max_subtree_weight == 7.5

    # Increment non-existing
    new_f2 = tree.increment_frequency("banana", delta=10.0)
    assert new_f2 == 10.0
    assert tree.contains("banana")
    assert tree.root.max_subtree_weight == 10.0


def test_minimal_deletion():
    tree = RadixTree()
    tree.insert("car", frequency=10.0)
    tree.insert("cart", frequency=20.0)
    tree.insert("carpet", frequency=30.0)
    initial_nodes = tree.node_count

    assert len(tree) == 3
    assert tree.delete("cart")
    assert len(tree) == 2
    assert not tree.contains("cart")
    assert tree.contains("car")
    assert tree.contains("carpet")
    # Per minimal deletion specification, edges are tombstoned without re-merging
    assert tree.node_count == initial_nodes

    # Non-existent delete
    assert not tree.delete("unknown")


def test_top_k_ordering():
    tree = RadixTree()
    items = [
        ("algorithm", 50.0),
        ("algorithmic", 10.0),
        ("algebra", 90.0),
        ("alias", 20.0),
        ("alien", 80.0),
    ]
    for w, s in items:
        tree.insert(w, frequency=s)

    # Prefix 'al' Top-3
    top3 = tree.top_k("al", k=3)
    assert len(top3) == 3
    assert top3[0][0] == "algebra"
    assert top3[0][1] == 90.0
    assert top3[1][0] == "alien"
    assert top3[1][1] == 80.0
    assert top3[2][0] == "algorithm"
    assert top3[2][1] == 50.0

    # Prefix 'alg' Top-2
    top_alg = tree.top_k("alg", k=2)
    assert len(top_alg) == 2
    assert top_alg[0][0] == "algebra"
    assert top_alg[1][0] == "algorithm"
