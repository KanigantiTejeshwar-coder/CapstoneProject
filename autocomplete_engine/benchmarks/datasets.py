"""Multilingual dataset generation and labeled typo benchmark corpora.

Generates multilingual terms across Latin, Devanagari, Han, Arabic, and Cyrillic,
simulating Zipfian frequency distributions P(r) ~ 1/r^s.
Produces labeled typo test cases for accurate MRR/Precision@K evaluation.
"""

import random
import unicodedata
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class MultilingualDatasetGenerator:
    """Generates synthetic and realistic multilingual corpora with Zipfian distributions."""

    # Seed vocabularies for realistic multilingual representation
    SEEDS = {
        "latin": [
            "algorithm", "autocomplete", "application", "architecture", "analysis",
            "backend", "binary", "benchmark", "branch", "bounded",
            "compression", "computer", "concurrent", "component", "cloud",
            "database", "dynamic", "dictionary", "distributed", "distance",
            "engine", "efficiency", "execution", "evaluation", "endpoint",
            "framework", "frequency", "frontend", "fuzzy", "function",
            "graph", "gradient", "garbage", "gateway", "generator",
            "hash", "heap", "hierarchy", "hardware", "hypertext",
            "index", "insertion", "interface", "iteration", "infrastructure",
            "javascript", "json", "journal", "junction", "kernel",
            "latency", "levenshtein", "lexicon", "language", "linear",
            "memory", "multilingual", "matcher", "matrix", "microservice",
            "network", "node", "normalization", "number", "null",
            "optimization", "operation", "output", "object", "overflow",
            "prefix", "priority", "performance", "patricia", "pipeline",
            "query", "queue", "quicksort", "quantum", "radix",
            "retrieval", "recursive", "routing", "runtime", "regression",
            "search", "string", "structure", "suggestion", "streaming",
            "trie", "throughput", "tree", "traversal", "tokenization",
            "unicode", "update", "unit", "utility", "unified",
            "vector", "verification", "virtual", "variable", "vocabulary",
            "weight", "window", "worker", "workflow", "websocket"
        ],
        "telugu": [
            "నమస్కారం", "భారతదేశం", "కంప్యూటర్", "అల్గోరిథం", "సమాచారం", "సాంకేతికత",
            "డేటా", "వ్యవస్థ", "శోధన", "పుస్తకం", "భాష", "విద్య", "విజ్ఞానం",
            "కార్యక్రమం", "నిల్వ", "నెట్‌వర్క్", "సాఫ్ట్‌వేర్", "పరిశోధన", "అభివృద్ధి",
            "పరికరాలు", "ఇంజనీరింగ్", "గణితం", "మేధస్సు", "వేగం", "ప్రక్రియ"
        ],
        "tamil": [
            "வணக்கம்", "இந்தியா", "கணினி", "வழிமுறை", "தரவு", "தொழில்நுட்பம்",
            "தேடல்", "மொழி", "புத்தகம்", "தகவல்", "அமைப்பு", "கல்வி", "அறிவு",
            "மென்பொருள்", "வலைப்பின்னல்", "நிரல்", "செயலி", "ஆராய்ச்சி", "வளர்ச்சி",
            "வேகம்", "செயற்கை நுண்ணறிவு", "அறிவியல்", "இயந்திரம்", "கருவி", "முன்னேற்றம்"
        ],
        "Hindi": [
            "नमस्ते", "भारत", "संगणक", "तकनीक", "डेटा", "ऐल्गोरिदम", "विकास",
            "मशीन", "भाषा", "पुस्तक", "शिक्षा", "विद्या", "खोज", "सूचना",
            "संरचना", "प्रणाली", "यंत्र", "गति", "सॉफ्टवेयर", "स्मृति",
            "जाल", "अनुसंधान", "प्रोग्राम", "सर्वर", "अंक", "ज्ञान"
        ]
    }

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def generate_vocabulary(
        self,
        total_words: int = 10000,
        zipf_s: float = 1.05
    ) -> List[Tuple[str, float]]:
        """Generates a vocabulary of specified size strictly across English, Telugu, Tamil, and Hindi."""

        words = set()

        # Add all base seeds first
        for script_words in self.SEEDS.values():
            for w in script_words:
                words.add(w)

        # Proportions: English 25%, Telugu 25%, Tamil 25%, Hindi 25%
        script_weights = [
            ("latin", 0.25),
            ("telugu", 0.25),
            ("tamil", 0.25),
            ("Hindi", 0.25)
        ]

        suffixes = [
            "ing", "tion", "ed", "er", "able", "ity",
            "ize", "ly", "ment", "ness", "s", "es"
        ]

        while len(words) < total_words:
            script_choice = self.rng.choices(
                [s[0] for s in script_weights],
                weights=[s[1] for s in script_weights]
            )[0]

            base_list = self.SEEDS[script_choice]
            w1 = self.rng.choice(base_list)

            if script_choice == "latin":
                # Create variations or compound terms for English
                coin = self.rng.random()

                if coin < 0.4:
                    w2 = self.rng.choice(base_list)
                    candidate = f"{w1}_{w2}"

                elif coin < 0.7:
                    sfx = self.rng.choice(suffixes)
                    candidate = f"{w1}{sfx}"

                else:
                    candidate = f"{w1}{self.rng.randint(1, 9999)}"

            else:
                # Telugu, Tamil, or Hindi compound terms or numbered queries
                coin = self.rng.random()

                if coin < 0.5:
                    w2 = self.rng.choice(base_list)
                    candidate = f"{w1}_{w2}"

                else:
                    candidate = f"{w1}_{self.rng.randint(1, 999)}"

            words.add(candidate)

        # Sort the set before slicing to guarantee deterministic ordering
        # across Python processes when using the same random seed.
        words_list = sorted(words)[:total_words]

        # Generate Zipfian popularity scores: P(r) = C / (r^s)
        # Ranks 1 to N
        ranks = np.arange(
            1,
            len(words_list) + 1,
            dtype=np.float64
        )

        raw_weights = 1.0 / np.power(
            ranks,
            zipf_s
        )

        # Scale into realistic query frequency range
        # (e.g. 10 to 1,000,000)
        norm_weights = (
            raw_weights / raw_weights.sum()
        ) * (total_words * 50) + 1.0

        # Shuffle words and assign ranks
        self.rng.shuffle(words_list)

        return [
            (word, float(norm_weights[i]))
            for i, word in enumerate(words_list)
        ]

    def generate_typo_benchmark_set(
        self,
        vocabulary: List[Tuple[str, float]],
        samples_per_bucket: int = 50
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generates labeled test cases partitioned into distinct typo buckets.

        Buckets:
            1. 'exact_prefix': No typos, query is a 3-5 character prefix of target.
            2. '1_edit': Single insertion, deletion, or substitution in prefix.
            3. '2_edit': Two edits in prefix.
            4. 'transposition': Single adjacent character swap.

        Returns:
            Dict mapping bucket_name to list of test case dicts:
            {"query": str, "target": str, "target_score": float, "bucket": str}
        """

        # Focus typo benchmark primarily on Latin/alphabetic words
        # where keyboard typos occur.
        latin_words = [
            item
            for item in vocabulary
            if item[0].isascii() and len(item[0]) >= 6
        ]

        if len(latin_words) < samples_per_bucket * 2:
            latin_words = [
                item
                for item in vocabulary
                if len(item[0]) >= 6
            ]

        sorted_latin = sorted(
            latin_words,
            key=lambda x: x[1],
            reverse=True
        )

        # Sample targets from upper quantile
        # (top frequent entities)
        pool_size = min(
            len(sorted_latin),
            max(
                samples_per_bucket * 8,
                int(len(sorted_latin) * 0.05)
            )
        )

        candidate_pool = sorted_latin[:pool_size]

        selected = self.rng.sample(
            candidate_pool,
            min(
                len(candidate_pool),
                samples_per_bucket * 4
            )
        )

        buckets = {
            "exact_prefix": [],
            "1_edit": [],
            "2_edit": [],
            "transposition": []
        }

        # 1. Exact Prefixes
        for word, score in selected[:samples_per_bucket]:
            prefix_len = min(
                len(word) - 1,
                self.rng.randint(3, 5)
            )

            prefix = word[:prefix_len]

            buckets["exact_prefix"].append({
                "query": prefix,
                "target": word,
                "target_score": score,
                "bucket": "exact_prefix"
            })

        # 2. 1-Edit Typos
        for word, score in selected[
            samples_per_bucket:samples_per_bucket * 2
        ]:
            prefix_len = min(
                len(word) - 1,
                self.rng.randint(4, 6)
            )

            prefix = list(word[:prefix_len])

            op = self.rng.choice(
                ["insert", "delete", "substitute"]
            )

            idx = self.rng.randint(
                1,
                len(prefix) - 1
            )

            if op == "insert":
                prefix.insert(
                    idx,
                    self.rng.choice(
                        "abcdefghijklmnopqrstuvwxyz"
                    )
                )

            elif op == "delete" and len(prefix) > 2:
                prefix.pop(idx)

            elif op == "substitute":
                ch = prefix[idx]

                replacement = self.rng.choice(
                    [
                        c
                        for c in "abcdefghijklmnopqrstuvwxyz"
                        if c != ch
                    ]
                )

                prefix[idx] = replacement

            buckets["1_edit"].append({
                "query": "".join(prefix),
                "target": word,
                "target_score": score,
                "bucket": "1_edit"
            })

        # 3. 2-Edit Typos
        for word, score in selected[
            samples_per_bucket * 2:samples_per_bucket * 3
        ]:
            prefix_len = min(
                len(word) - 1,
                self.rng.randint(5, 7)
            )

            prefix = list(word[:prefix_len])

            # Apply two sequential edits
            for _ in range(2):
                idx = self.rng.randint(
                    1,
                    max(1, len(prefix) - 1)
                )

                op = self.rng.choice(
                    ["substitute", "insert"]
                )

                if op == "insert":
                    prefix.insert(
                        idx,
                        self.rng.choice(
                            "abcdefghijklmnopqrstuvwxyz"
                        )
                    )

                else:
                    prefix[idx] = self.rng.choice(
                        "abcdefghijklmnopqrstuvwxyz"
                    )

            buckets["2_edit"].append({
                "query": "".join(prefix),
                "target": word,
                "target_score": score,
                "bucket": "2_edit"
            })

        # 4. Transpositions
        for word, score in selected[
            samples_per_bucket * 3:samples_per_bucket * 4
        ]:
            prefix_len = min(
                len(word) - 1,
                self.rng.randint(4, 6)
            )

            prefix = list(word[:prefix_len])

            if len(prefix) >= 3:
                idx = self.rng.randint(
                    1,
                    len(prefix) - 2
                )

                prefix[idx], prefix[idx + 1] = (
                    prefix[idx + 1],
                    prefix[idx]
                )

            buckets["transposition"].append({
                "query": "".join(prefix),
                "target": word,
                "target_score": score,
                "bucket": "transposition"
            })

        return buckets