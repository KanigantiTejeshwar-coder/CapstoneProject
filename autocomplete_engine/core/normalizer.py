"""Multilingual text normalization and script routing for English, Telugu, Tamil, and Hindi.

Supports Unicode NFC/NFKC normalization, case-folding, Latin diacritic-stripping,
and Indian/English script classification.
"""

import unicodedata
from enum import Enum


class ScriptType(str, Enum):
    LATIN = "latin"            # English
    TELUGU = "telugu"          # Telugu
    TAMIL = "tamil"            # Tamil
    DEVANAGARI = "devanagari"  # Hindi
    OTHER = "other"


class UnicodeNormalizer:
    """Handles text normalization and script routing for English, Telugu, Tamil, and Hindi."""

    def __init__(self, strip_accents: bool = False, lowercase: bool = True) -> None:
        self.strip_accents = strip_accents
        self.lowercase = lowercase

    def normalize(self, text: str) -> str:
        """Normalizes Unicode text using NFKC, optional case-folding and accent stripping."""
        if not text:
            return ""

        # Step 1: NFKC normalization
        norm = unicodedata.normalize("NFKC", text)

        # Step 2: Optional case folding
        if self.lowercase:
            norm = norm.casefold()

        # Step 3: Optional accent stripping (specifically for Latin diacritical marks)
        # Keeps Telugu, Tamil, and Hindi vowel signs, viramas, and pulli completely intact
        if self.strip_accents:
            decomposed = unicodedata.normalize("NFD", norm)
            stripped = "".join(
                ch for ch in decomposed
                if not (0x0300 <= ord(ch) <= 0x036F and unicodedata.category(ch) == "Mn")
            )
            norm = unicodedata.normalize("NFC", stripped)

        return norm.strip()

    @staticmethod
    def detect_script(text: str) -> ScriptType:
        """Detects the predominant script of a given query or token among English, Telugu, Tamil, Hindi."""
        if not text:
            return ScriptType.OTHER

        script_counts = {
            ScriptType.LATIN: 0,
            ScriptType.TELUGU: 0,
            ScriptType.TAMIL: 0,
            ScriptType.DEVANAGARI: 0,
            ScriptType.OTHER: 0,
        }

        for ch in text:
            if ch.isspace() or unicodedata.category(ch).startswith("P"):
                continue

            cp = ord(ch)
            # English / Latin: Basic Latin, Latin-1, Extended-A/B
            if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F) or (0x1E00 <= cp <= 0x1EFF):
                script_counts[ScriptType.LATIN] += 1
            # Telugu (U+0C00 to U+0C7F)
            elif 0x0C00 <= cp <= 0x0C7F:
                script_counts[ScriptType.TELUGU] += 1
            # Tamil (U+0B80 to U+0BFF)
            elif 0x0B80 <= cp <= 0x0BFF:
                script_counts[ScriptType.TAMIL] += 1
            # Hindi / Devanagari (U+0900 to U+097F)
            elif 0x0900 <= cp <= 0x097F or 0xA8E0 <= cp <= 0xA8FF:
                script_counts[ScriptType.DEVANAGARI] += 1
            else:
                script_counts[ScriptType.OTHER] += 1

        total = sum(script_counts.values())
        if total == 0:
            return ScriptType.OTHER

        predominant_script, _ = max(script_counts.items(), key=lambda item: item[1])
        return predominant_script
