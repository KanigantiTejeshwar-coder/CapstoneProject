"""Compressed Trie (Radix Tree) with in-place edge splitting and Top-K pruning.

Provides O(|word|) insertion, dynamic frequency updates without index rebuild,
minimal deletion (tombstoning), and branch-and-bound Top-K search using
cached `max_subtree_weight`.
"""

import heapq
import threading
from functools import wraps
from typing import List, Tuple, Optional, Dict, Any, Callable

from autocomplete_engine.core.node import RadixNode


def synchronized(method: Callable) -> Callable:
    """Protect a RadixTree method with the tree's re-entrant lock."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class RadixTree:
    """Radix Tree (Compressed Trie) implementation."""

    def __init__(self) -> None:
        self.root: RadixNode = RadixNode(
            edge_label="",
            is_terminal=False,
            frequency=0.0
        )
        self._size: int = 0
        self._node_count: int = 1
        self._lock = threading.RLock()

    def __getstate__(self):
        """Return pickle-safe state without the runtime lock."""
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state):
        """Recreate lock after unpickling."""
        self.__dict__.update(state)
        self._lock = threading.RLock()

    @synchronized
    def __len__(self) -> int:
        return self._size

    @property
    @synchronized
    def node_count(self) -> int:
        """Returns the current number of nodes under the tree lock."""
        return self._node_count

    @synchronized
    def insert(
        self,
        word: str,
        frequency: float = 1.0,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Inserts a word into the Radix Tree.

        Performs in-place edge splitting without rebuilding the tree.
        Propagates `max_subtree_weight` up the ancestor chain.

        Returns:
            True if word was newly inserted, False if word existed and was updated.
        """
        if not word:
            # Handle empty string at root
            is_new = not self.root.is_terminal
            self.root.is_terminal = True
            self.root.frequency = max(self.root.frequency, frequency)

            if data is not None:
                self.root.data = data

            self.root.recompute_max_weight()

            if is_new:
                self._size += 1

            return is_new

        ancestor_path: List[RadixNode] = [self.root]
        curr = self.root
        s = word
        is_new_word = True

        while s:
            first_ch = s[0]

            if first_ch not in curr.children:
                # No common edge: attach new leaf directly
                new_leaf = RadixNode(
                    edge_label=s,
                    is_terminal=True,
                    frequency=frequency,
                    data=data
                )

                curr.children[first_ch] = new_leaf
                self._node_count += 1
                self._size += 1
                ancestor_path.append(new_leaf)
                break

            child = curr.children[first_ch]
            label = child.edge_label

            # Determine length of common prefix between label and s
            common_len = 0
            limit = min(len(label), len(s))

            while common_len < limit and label[common_len] == s[common_len]:
                common_len += 1

            if common_len == len(label):
                # The entire edge matches a prefix of s
                curr = child
                ancestor_path.append(curr)
                s = s[common_len:]

                if not s:
                    # Word ends exactly at this existing node
                    is_new_word = not curr.is_terminal
                    curr.is_terminal = True
                    curr.frequency = max(curr.frequency, frequency)

                    if data is not None:
                        curr.data = data

                    if is_new_word:
                        self._size += 1

                    break

            else:
                # Partial match: edge split required

                # 1. Create intermediate node for the common prefix
                common_prefix = label[:common_len]
                remaining_child_label = label[common_len:]

                split_node = RadixNode(
                    edge_label=common_prefix,
                    is_terminal=False,
                    frequency=0.0
                )

                self._node_count += 1

                # 2. Update existing child to represent remaining edge
                child.edge_label = remaining_child_label
                split_node.children[remaining_child_label[0]] = child
                curr.children[first_ch] = split_node

                ancestor_path.append(split_node)

                # 3. Check if inserted word terminates at split_node or branches
                if common_len == len(s):
                    split_node.is_terminal = True
                    split_node.frequency = frequency
                    split_node.data = data
                    self._size += 1
                    split_node.recompute_max_weight()

                else:
                    # Create new leaf for remaining suffix of s
                    remaining_s = s[common_len:]

                    new_leaf = RadixNode(
                        edge_label=remaining_s,
                        is_terminal=True,
                        frequency=frequency,
                        data=data
                    )

                    split_node.children[remaining_s[0]] = new_leaf
                    self._node_count += 1
                    self._size += 1
                    split_node.recompute_max_weight()
                    ancestor_path.append(new_leaf)

                break

        # Upward propagation of max_subtree_weight
        for node in reversed(ancestor_path):
            node.recompute_max_weight()

        return is_new_word

    @synchronized
    def increment_frequency(self, word: str, delta: float = 1.0) -> float:
        """Increments word frequency dynamically. If word doesn't exist, inserts it."""
        if not word:
            self.root.is_terminal = True
            self.root.frequency += delta
            self.root.recompute_max_weight()
            return self.root.frequency

        ancestor_path: List[RadixNode] = [self.root]
        curr = self.root
        s = word

        while s:
            first_ch = s[0]

            if first_ch not in curr.children:
                # Word doesn't exist yet, insert with delta
                self.insert(word, frequency=delta)
                return delta

            child = curr.children[first_ch]
            label = child.edge_label

            common_len = 0
            limit = min(len(label), len(s))

            while common_len < limit and label[common_len] == s[common_len]:
                common_len += 1

            if common_len == len(label):
                curr = child
                ancestor_path.append(curr)
                s = s[common_len:]

                if not s:
                    if not curr.is_terminal:
                        curr.is_terminal = True
                        curr.frequency = delta
                        self._size += 1
                    else:
                        curr.frequency += delta

                    # Upward propagation
                    for node in reversed(ancestor_path):
                        node.recompute_max_weight()

                    return curr.frequency

            else:
                # Partial match -> word doesn't exist yet
                self.insert(word, frequency=delta)
                return delta

        return delta

    @synchronized
    def delete(self, word: str) -> bool:
        """Tombstones the terminal flag of a word.

        Per design decision: minimal deletion clears the terminal flag
        and decrements word count, without re-merging edges.
        """
        if not word:
            if self.root.is_terminal:
                self.root.is_terminal = False
                self.root.frequency = 0.0
                self.root.recompute_max_weight()
                self._size -= 1
                return True

            return False

        ancestor_path: List[RadixNode] = [self.root]
        curr = self.root
        s = word

        while s:
            first_ch = s[0]

            if first_ch not in curr.children:
                return False

            child = curr.children[first_ch]
            label = child.edge_label

            if not s.startswith(label):
                return False

            curr = child
            ancestor_path.append(curr)
            s = s[len(label):]

        if curr.is_terminal:
            curr.is_terminal = False
            curr.frequency = 0.0
            self._size -= 1

            for node in reversed(ancestor_path):
                node.recompute_max_weight()

            return True

        return False

    @synchronized
    def contains(self, word: str) -> bool:
        """Checks whether an exact word exists in the Radix Tree."""
        if not word:
            return self.root.is_terminal

        curr = self.root
        s = word

        while s:
            first_ch = s[0]

            if first_ch not in curr.children:
                return False

            child = curr.children[first_ch]
            label = child.edge_label

            if not s.startswith(label):
                return False

            curr = child
            s = s[len(label):]

        return curr.is_terminal

    @synchronized
    def get_frequency(self, word: str) -> Optional[float]:
        """Returns frequency score of word if present, otherwise None."""
        if not word:
            return self.root.frequency if self.root.is_terminal else None

        curr = self.root
        s = word

        while s:
            first_ch = s[0]

            if first_ch not in curr.children:
                return None

            child = curr.children[first_ch]
            label = child.edge_label

            if not s.startswith(label):
                return None

            curr = child
            s = s[len(label):]

        return curr.frequency if curr.is_terminal else None

    @synchronized
    def search_prefix_node(
        self,
        prefix: str
    ) -> Optional[Tuple[RadixNode, str]]:
        """Walks to the node covering prefix.

        Returns:
            Tuple of (matched_node, path_string_accumulated) or None if prefix not found.
        """
        if not prefix:
            return self.root, ""

        curr = self.root
        s = prefix
        accumulated: List[str] = []

        while s:
            first_ch = s[0]

            if first_ch not in curr.children:
                return None

            child = curr.children[first_ch]
            label = child.edge_label

            if len(s) <= len(label):
                # The remaining prefix is shorter than or equal to edge label
                if label.startswith(s):
                    # Found! The child covers this prefix
                    accumulated.append(label)
                    return child, "".join(accumulated)
                else:
                    return None

            else:
                # Remaining prefix is longer than edge label
                if not s.startswith(label):
                    return None

                accumulated.append(label)
                curr = child
                s = s[len(label):]

        return curr, "".join(accumulated)

    @synchronized
    def top_k(
        self,
        prefix: str,
        k: int = 5
    ) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        """Finds Top-K highest-frequency completions for given prefix.

        Uses best-first search with branch-and-bound pruning prioritized by
        `max_subtree_weight`.

        Returns:
            List of (completed_word, frequency, data) tuples sorted descending by frequency.
        """
        if k <= 0:
            return []

        match_res = self.search_prefix_node(prefix)

        if match_res is None:
            return []

        start_node, matched_path = match_res

        # Bounded Priority Queue / Best-First Search
        # Heap elements: (-node.max_subtree_weight, word_so_far, node)
        heap: List[Tuple[float, str, RadixNode]] = []

        heapq.heappush(
            heap,
            (-start_node.max_subtree_weight, matched_path, start_node)
        )

        results: List[
            Tuple[str, float, Optional[Dict[str, Any]]]
        ] = []

        while heap:
            neg_max_w, word_acc, node = heapq.heappop(heap)
            max_w = -neg_max_w

            # Branch-and-bound early termination
            if len(results) >= k and max_w <= results[-1][1]:
                break

            if node.is_terminal:
                # Insert into results maintaining sorted descending order
                results.append(
                    (word_acc, node.frequency, node.data)
                )

                results.sort(
                    key=lambda item: item[1],
                    reverse=True
                )

                if len(results) > k:
                    results.pop()

            for child in node.children.values():
                # Prune child if it can't beat current k-th score
                if (
                    len(results) >= k
                    and child.max_subtree_weight <= results[-1][1]
                ):
                    continue

                heapq.heappush(
                    heap,
                    (
                        -child.max_subtree_weight,
                        word_acc + child.edge_label,
                        child
                    )
                )

        return results[:k] 