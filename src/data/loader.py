
import re
import hashlib
from typing import List, Tuple

from sklearn.datasets import fetch_20newsgroups

from src.config import MAX_CHARS_PER_DOC, MIN_CHARS_PER_DOC



_RE_QUOTES  = re.compile(r"^>.*$", re.MULTILINE)

_RE_HEADERS = re.compile(
    r"^(From|Subject|Organization|Lines|Message-ID|NNTP-Posting-Host|"
    r"Distribution|Newsgroups|Path|Date|Xref|Article-I\.D\.|Reply-To|"
    r"Summary|Keywords|Expires|Followup-To|Sender|References)[^\n]*\n",
    re.MULTILINE | re.IGNORECASE,
)

_RE_EMAIL   = re.compile(r"\S+@\S+\.\S+")

_RE_PUNCT   = re.compile(r"[^\w\s]")

_RE_SPACE   = re.compile(r"\s+")


def _clean(text: str) -> str:
    """Return a cleaned, truncated version of a raw newsgroup post."""
    text = _RE_HEADERS.sub("", text)
    text = _RE_QUOTES.sub("", text)
    text = _RE_EMAIL.sub(" ", text)
    text = _RE_PUNCT.sub(" ", text)
    text = _RE_SPACE.sub(" ", text).strip()
    return text[:MAX_CHARS_PER_DOC]


def load_corpus(
    subset: str = "all",
) -> Tuple[List[str], List[int], List[str]]:
    """
    Fetch, clean, and deduplicate the 20 Newsgroups corpus.

    Parameters
    ----------
    subset : "train" | "test" | "all"

    Returns
    -------
    texts     : cleaned document strings
    labels    : integer category ids (0-19)
    ids       : stable SHA-256 document ids (hex, 16 chars)
    """
    raw = fetch_20newsgroups(
        subset=subset,
        remove=("headers", "footers", "quotes"),  
        shuffle=True,
        random_state=42,
    )

    texts, labels, ids = [], [], []
    seen_hashes: set = set()

    for doc, label in zip(raw.data, raw.target):
        cleaned = _clean(doc)

        if len(cleaned) < MIN_CHARS_PER_DOC:
            continue

        doc_hash = hashlib.sha256(cleaned.encode()).hexdigest()[:16]
        if doc_hash in seen_hashes:
            continue
        seen_hashes.add(doc_hash)

        texts.append(cleaned)
        labels.append(int(label))
        ids.append(doc_hash)

    print(
        f"[loader] Loaded {len(texts):,} documents "
        f"({len(raw.data) - len(texts):,} dropped as short/duplicate)"
    )
    return texts, labels, ids


def category_names() -> List[str]:
    """Return the 20 newsgroup category names in label order."""
    raw = fetch_20newsgroups(subset="train")
    return list(raw.target_names)
