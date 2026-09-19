
from typing import List, Tuple, Optional, Dict, Any
from autocomplete_engine.core.radix_tree import RadixTree
from autocomplete_engine.core.node import RadixNode


class LevenshteinRadixMatcher:
    def __init__(
        self,
        max_edits: int = 2,
        allow_transpositions: bool = True
    ) -> None:
        self.max_edits = max_edits
        self.allow_transpositions = allow_transpositions
        self.nodes_visited = 0  # Telemetry for verification tests

    def search(
        self,
        tree: RadixTree,
        query: str,
        k: int = 5,
        prefix_mode: bool = True
    ) -> List[Tuple[str, float, int, Optional[Dict[str, Any]]]]:
        self.nodes_visited = 0

        if not query:
            # Empty query returns exact top-k from root
            return [
                (w, score, 0, data)
                for w, score, data in tree.top_k("", k=k)
            ]

        initial_row = list(range(len(query) + 1))
        candidates: List[
            Tuple[str, float, int, Optional[Dict[str, Any]]]
        ] = []
        with tree._lock:
            for child in tree.root.children.values():
                self._search_recursive(
                    node=child,
                    accumulated="",
                    edge_label=child.edge_label,
                    query=query,
                    prev_row=initial_row,
                    prev_prev_row=None,
                    prev_char="",
                    candidates=candidates,
                    prefix_mode=prefix_mode,
                    k=k
                )
        candidates.sort(key=lambda item: (item[2], -item[1]))

        # Deduplicate words keeping lowest edit distance and highest score
        seen = set()
        deduped = []

        for word, score, dist, data in candidates:
            if word not in seen:
                seen.add(word)
                deduped.append(
                    (word, score, dist, data)
                )

                if len(deduped) >= k:
                    break

        return deduped

    def _search_recursive(
        self,
        node: RadixNode,
        accumulated: str,
        edge_label: str,
        query: str,
        prev_row: List[int],
        prev_prev_row: Optional[List[int]],
        prev_char: str,
        candidates: List[
            Tuple[str, float, int, Optional[Dict[str, Any]]]
        ],
        prefix_mode: bool,
        k: int
    ) -> None:
        self.nodes_visited += 1

        current_row = prev_row
        row_history = [prev_row]

        if prev_prev_row is not None:
            row_history.insert(0, prev_prev_row)

        curr_accum = accumulated
        curr_prev_char = prev_char

        prefix_match_dist: Optional[int] = None

        # Advance DP row for each character in the compressed edge label
        for ch in edge_label:
            curr_accum += ch
            new_row = [current_row[0] + 1]

            for i in range(1, len(query) + 1):
                q_ch = query[i - 1]

                insert_cost = current_row[i] + 1
                delete_cost = new_row[i - 1] + 1
                replace_cost = current_row[i - 1] + (
                    0 if ch == q_ch else 1
                )

                cost = min(
                    insert_cost,
                    delete_cost,
                    replace_cost
                )

                # Damerau-Levenshtein transposition check
                if (
                    self.allow_transpositions
                    and len(row_history) >= 2
                    and i > 1
                    and ch == query[i - 2]
                    and curr_prev_char == q_ch
                    and curr_prev_char != ""
                ):
                    trans_cost = row_history[-2][i - 2] + 1
                    cost = min(cost, trans_cost)

                new_row.append(cost)

            row_history.append(new_row)

            if len(row_history) > 3:
                row_history.pop(0)

            current_row = new_row
            curr_prev_char = ch

            # Check if full query is consumed at this character
            # (prefix match)
            if prefix_mode and new_row[-1] <= self.max_edits:
                if (
                    prefix_match_dist is None
                    or new_row[-1] < prefix_match_dist
                ):
                    prefix_match_dist = new_row[-1]

        # Case 1: Fuzzy Prefix Match along this edge
        if prefix_mode and prefix_match_dist is not None:
            if node.is_terminal:
                candidates.append(
                    (
                        curr_accum,
                        node.frequency,
                        prefix_match_dist,
                        node.data
                    )
                )

            self._collect_completions(
                node,
                curr_accum,
                prefix_match_dist,
                candidates
            )

            return

       
        if min(current_row) > self.max_edits:
            return

        # Case 2: Exact / Fuzzy Full-word Match at this node
        final_dist = current_row[-1]

        if node.is_terminal and final_dist <= self.max_edits:
            candidates.append(
                (
                    curr_accum,
                    node.frequency,
                    final_dist,
                    node.data
                )
            )

        for child in node.children.values():
            self._search_recursive(
                node=child,
                accumulated=curr_accum,
                edge_label=child.edge_label,
                query=query,
                prev_row=current_row,
                prev_prev_row=(
                    row_history[-2]
                    if len(row_history) >= 2
                    else None
                ),
                prev_char=curr_prev_char,
                candidates=candidates,
                prefix_mode=prefix_mode,
                k=k
            )

    def _collect_completions(
        self,
        node: RadixNode,
        prefix_acc: str,
        edit_distance: int,
        candidates: List[
            Tuple[str, float, int, Optional[Dict[str, Any]]]
        ]
    ) -> None:
        """Collects terminal completions from this node's subtree."""

        # Called from within the locked search traversal.
        for child in node.children.values():
            child_word = prefix_acc + child.edge_label

            if child.is_terminal:
                candidates.append(
                    (
                        child_word,
                        child.frequency,
                        edit_distance,
                        child.data
                    )
                )

            self._collect_completions(
                child,
                child_word,
                edit_distance,
                candidates
            )
