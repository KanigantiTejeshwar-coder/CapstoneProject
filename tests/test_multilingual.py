"""Unit tests for multilingual normalization and script routing for English, Telugu, Tamil, and Hindi."""

import pytest
from autocomplete_engine.core.normalizer import UnicodeNormalizer, ScriptType
from autocomplete_engine.core.engine import AutocompleteEngine


def test_script_detection_exclusive_languages():
    """Confirms script detection for English, Telugu, Tamil, and Hindi."""
    normalizer = UnicodeNormalizer()
    assert normalizer.detect_script("algorithm") == ScriptType.LATIN
    assert normalizer.detect_script("computer") == ScriptType.LATIN
    assert normalizer.detect_script("నమస్కారం") == ScriptType.TELUGU
    assert normalizer.detect_script("భారతదేశం") == ScriptType.TELUGU
    assert normalizer.detect_script("வணக்கம்") == ScriptType.TAMIL
    assert normalizer.detect_script("இந்தியா") == ScriptType.TAMIL
    assert normalizer.detect_script("नमस्ते") == ScriptType.DEVANAGARI
    assert normalizer.detect_script("भारत") == ScriptType.DEVANAGARI


def test_unicode_normalization_preserves_indic_scripts():
    """Confirms that Unicode normalization preserves Telugu, Tamil, and Hindi vowel signs and viramas."""
    norm = UnicodeNormalizer(strip_accents=True, lowercase=True)
    # Latin lowercasing & diacritic stripping
    assert norm.normalize("Café") == "cafe"
    assert norm.normalize("ALGORITHM") == "algorithm"
    # Telugu NFKC preserves consonants, vowel signs, and viramas
    assert norm.normalize("నమస్కారం") == "నమస్కారం"
    assert norm.normalize("భారతదేశం") == "భారతదేశం"
    # Tamil NFKC preserves pulli and vowel signs
    assert norm.normalize("வணக்கம்") == "வணக்கம்"
    assert norm.normalize("இந்தியா") == "இந்தியா"
    # Hindi NFKC preserves matras and conjuncts
    assert norm.normalize("नमस्ते") == "नमस्ते"
    assert norm.normalize("भारत") == "भारत"


def test_multilingual_routing_and_search_all_four_languages():
    """Confirms accurate Top-K autocomplete search across English, Telugu, Tamil, and Hindi."""
    engine = AutocompleteEngine()
    engine.insert("algorithm", score=10.0)
    engine.insert("నమస్కారం", score=20.0)
    engine.insert("வணக்கம்", score=30.0)
    engine.insert("नमस्ते", score=40.0)

    # 1. English / Latin prefix
    res_en = engine.search("alg", k=5)
    assert len(res_en) == 1
    assert res_en[0]["word"] == "algorithm"
    assert res_en[0]["script"] == "latin"

    # 2. Telugu prefix
    res_te = engine.search("నమస్", k=5)
    assert len(res_te) == 1
    assert res_te[0]["word"] == "నమస్కారం"
    assert res_te[0]["script"] == "telugu"

    # 3. Tamil prefix
    res_ta = engine.search("வணக்", k=5)
    assert len(res_ta) == 1
    assert res_ta[0]["word"] == "வணக்கம்"
    assert res_ta[0]["script"] == "tamil"

    # 4. Hindi / Devanagari prefix
    res_hi = engine.search("नम", k=5)
    assert len(res_hi) == 1
    assert res_hi[0]["word"] == "नमस्ते"
    assert res_hi[0]["script"] == "devanagari"
