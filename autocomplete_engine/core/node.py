"""Memory-optimized Node structures for Radix Tree and Standard Trie.

Both node types maintain `max_subtree_weight` to ensure a fair comparison
when benchmarking top-k prefix searches.
"""

from typing import Dict, Any, Optional


class RadixNode:
    """Compressed Trie (Radix Tree) Node.
    
    Attributes:
        edge_label: Compressed substring corresponding to the incoming edge.
        children: Dict mapping the first character of each child's edge_label to that child node.
        is_terminal: True if this node represents the end of a valid word/phrase.
        frequency: Popularity/frequency score of the word ending at this node.
        max_subtree_weight: Maximum frequency among all terminal nodes in this subtree.
        data: Optional auxiliary metadata dictionary.
    """
    __slots__ = (
        'edge_label',
        'children',
        'is_terminal',
        'frequency',
        'max_subtree_weight',
        'data',
    )

    def __init__(
        self,
        edge_label: str = "",
        is_terminal: bool = False,
        frequency: float = 0.0,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        self.edge_label: str = edge_label
        self.children: Dict[str, RadixNode] = {}
        self.is_terminal: bool = is_terminal
        self.frequency: float = frequency
        self.max_subtree_weight: float = frequency if is_terminal else 0.0
        self.data: Optional[Dict[str, Any]] = data

    def recompute_max_weight(self) -> float:
        """Recomputes max_subtree_weight from this node and all direct children."""
        max_w = self.frequency if self.is_terminal else 0.0
        for child in self.children.values():
            if child.max_subtree_weight > max_w:
                max_w = child.max_subtree_weight
        self.max_subtree_weight = max_w
        return max_w

    def to_dict(self) -> Dict[str, Any]:
        """Serializes node and subtree to dictionary."""
        return {
            "label": self.edge_label,
            "term": self.is_terminal,
            "freq": self.frequency,
            "max_w": self.max_subtree_weight,
            "data": self.data,
            "children": {k: c.to_dict() for k, c in self.children.items()}
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RadixNode":
        """Deserializes node and subtree from dictionary."""
        node = cls(
            edge_label=d.get("label", ""),
            is_terminal=d.get("term", False),
            frequency=d.get("freq", 0.0),
            data=d.get("data")
        )
        node.max_subtree_weight = d.get("max_w", node.frequency if node.is_terminal else 0.0)
        for k, cd in d.get("children", {}).items():
            node.children[k] = cls.from_dict(cd)
        return node


class StandardTrieNode:
    """Standard uncompressed character-by-character Trie Node.
    
    Attributes:
        char: Single character on incoming transition (empty for root).
        children: Dict mapping single characters to child nodes.
        is_terminal: True if this node represents the end of a valid word.
        frequency: Frequency score of the word ending at this node.
        max_subtree_weight: Maximum frequency in this subtree (kept identical to
            RadixNode to isolate compression as the primary benchmark variable).
        data: Optional auxiliary metadata dictionary.
    """
    __slots__ = (
        'char',
        'children',
        'is_terminal',
        'frequency',
        'max_subtree_weight',
        'data',
    )

    def __init__(
        self,
        char: str = "",
        is_terminal: bool = False,
        frequency: float = 0.0,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        self.char: str = char
        self.children: Dict[str, StandardTrieNode] = {}
        self.is_terminal: bool = is_terminal
        self.frequency: float = frequency
        self.max_subtree_weight: float = frequency if is_terminal else 0.0
        self.data: Optional[Dict[str, Any]] = data

    def recompute_max_weight(self) -> float:
        """Recomputes max_subtree_weight from this node and all direct children."""
        max_w = self.frequency if self.is_terminal else 0.0
        for child in self.children.values():
            if child.max_subtree_weight > max_w:
                max_w = child.max_subtree_weight
        self.max_subtree_weight = max_w
        return max_w

    def to_dict(self) -> Dict[str, Any]:
        """Serializes node and subtree to dictionary."""
        return {
            "char": self.char,
            "term": self.is_terminal,
            "freq": self.frequency,
            "max_w": self.max_subtree_weight,
            "data": self.data,
            "children": {k: c.to_dict() for k, c in self.children.items()}
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StandardTrieNode":
        """Deserializes node and subtree from dictionary."""
        node = cls(
            char=d.get("char", ""),
            is_terminal=d.get("term", False),
            frequency=d.get("freq", 0.0),
            data=d.get("data")
        )
        node.max_subtree_weight = d.get("max_w", node.frequency if node.is_terminal else 0.0)
        for k, cd in d.get("children", {}).items():
            node.children[k] = cls.from_dict(cd)
        return node
