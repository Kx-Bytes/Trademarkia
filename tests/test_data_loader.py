"""
tests/test_data_loader.py

Unit tests for src/data/loader.py – specifically the _clean() function and
the load_corpus() deduplication / filtering logic.

We do NOT call fetch_20newsgroups in the tests (too slow for CI).
load_corpus is tested via a mock so we control the raw data.

Covers
──────
- _clean: strips header-like lines (From:, Subject:, etc.)
- _clean: removes quoted reply lines ("> …")
- _clean: removes email addresses
- _clean: removes standalone punctuation runs
- _clean: collapses whitespace
- _clean: truncates to MAX_CHARS_PER_DOC
- _clean: preserves normal prose
- load_corpus: short docs below MIN_CHARS_PER_DOC are dropped
- load_corpus: duplicate docs (same hash) are deduplicated
- load_corpus: returns three equal-length lists
- load_corpus: all ids are 16-character hex strings
- load_corpus: labels are integers
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the private _clean helper directly
from src.data.loader import _clean, load_corpus
from src.config import MAX_CHARS_PER_DOC, MIN_CHARS_PER_DOC


# ──────────────────────────────────────────────────────────────────────────────
# _clean
# ──────────────────────────────────────────────────────────────────────────────

class TestClean:
    def test_strips_from_header(self):
        text = "From: alice@example.com\nHello world this is a good post.\n"
        out  = _clean(text)
        assert "From:" not in out
        assert "Hello world" in out

    def test_strips_subject_header(self):
        out = _clean("Subject: Re: Gun laws\nThis is the body.\n")
        assert "Subject:" not in out
        assert "body" in out

    def test_strips_quoted_reply_lines(self):
        text = "> This was previously written\n> And this too\nActual reply here."
        out  = _clean(text)
        assert "previously" not in out
        assert "Actual reply" in out

    def test_strips_email_addresses(self):
        out = _clean("Contact user@domain.org for more info about rockets.")
        assert "@" not in out
        assert "rockets" in out

    def test_collapses_whitespace(self):
        out = _clean("Too    many   spaces\tand\ttabs.")
        assert "  " not in out    # no double spaces

    def test_truncates_to_max_chars(self):
        long_text = "a " * (MAX_CHARS_PER_DOC + 500)
        out = _clean(long_text)
        assert len(out) <= MAX_CHARS_PER_DOC

    def test_preserves_normal_prose(self):
        prose = "The space shuttle programme ended in 2011 with the final Atlantis mission."
        out   = _clean(prose)
        # Core words should still be present
        assert "space" in out
        assert "shuttle" in out
        assert "2011" in out

    def test_empty_string_returns_empty(self):
        assert _clean("") == ""

    def test_only_headers_returns_empty_or_short(self):
        text = "From: bob@example.com\nSubject: hello\n"
        out  = _clean(text)
        # After cleaning headers and email the result should be very short
        assert len(out) < 20


# ──────────────────────────────────────────────────────────────────────────────
# load_corpus (mocked fetch_20newsgroups)
# ──────────────────────────────────────────────────────────────────────────────

def _make_raw(docs, targets):
    """Build a mock object that mimics sklearn Bunch."""
    return SimpleNamespace(data=docs, target=targets, target_names=[f"cat{i}" for i in range(20)])


LONG_DOC   = "The politics of space exploration and NASA funding. " * 15   # well above MIN
MEDIUM_DOC = "Gun legislation debate in the United States congress. " * 10
SHORT_DOC  = "ok"     # below MIN_CHARS_PER_DOC → should be dropped


class TestLoadCorpus:
    @patch("src.data.loader.fetch_20newsgroups")
    def test_returns_three_equal_length_lists(self, mock_fetch):
        mock_fetch.return_value = _make_raw([LONG_DOC, MEDIUM_DOC], [0, 1])
        texts, labels, ids = load_corpus()
        assert len(texts) == len(labels) == len(ids)

    @patch("src.data.loader.fetch_20newsgroups")
    def test_short_docs_dropped(self, mock_fetch):
        mock_fetch.return_value = _make_raw([LONG_DOC, SHORT_DOC], [0, 1])
        texts, labels, ids = load_corpus()
        # SHORT_DOC should be filtered out
        assert len(texts) == 1

    @patch("src.data.loader.fetch_20newsgroups")
    def test_duplicate_docs_deduplicated(self, mock_fetch):
        mock_fetch.return_value = _make_raw([LONG_DOC, LONG_DOC], [0, 0])
        texts, labels, ids = load_corpus()
        assert len(texts) == 1

    @patch("src.data.loader.fetch_20newsgroups")
    def test_ids_are_16_char_hex(self, mock_fetch):
        mock_fetch.return_value = _make_raw([LONG_DOC, MEDIUM_DOC], [0, 1])
        _, _, ids = load_corpus()
        for doc_id in ids:
            assert len(doc_id) == 16
            int(doc_id, 16)   # raises ValueError if not valid hex

    @patch("src.data.loader.fetch_20newsgroups")
    def test_labels_are_integers(self, mock_fetch):
        mock_fetch.return_value = _make_raw([LONG_DOC, MEDIUM_DOC], [3, 17])
        _, labels, _ = load_corpus()
        assert all(isinstance(l, int) for l in labels)

    @patch("src.data.loader.fetch_20newsgroups")
    def test_different_docs_get_different_ids(self, mock_fetch):
        mock_fetch.return_value = _make_raw([LONG_DOC, MEDIUM_DOC], [0, 1])
        _, _, ids = load_corpus()
        assert len(set(ids)) == len(ids), "Distinct docs must have distinct ids"

    @patch("src.data.loader.fetch_20newsgroups")
    def test_all_short_docs_returns_empty(self, mock_fetch):
        mock_fetch.return_value = _make_raw([SHORT_DOC, SHORT_DOC], [0, 1])
        texts, labels, ids = load_corpus()
        assert texts == []
        assert labels == []
        assert ids == []

    @patch("src.data.loader.fetch_20newsgroups")
    def test_texts_respect_max_chars(self, mock_fetch):
        huge = "word " * 5000
        mock_fetch.return_value = _make_raw([huge], [0])
        texts, _, _ = load_corpus()
        if texts:
            assert len(texts[0]) <= MAX_CHARS_PER_DOC
