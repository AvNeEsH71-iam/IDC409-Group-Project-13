"""
Small text helpers shared by every module.

Planned contents:
- normalize_text(): lowercase, strip punctuation, collapse spaces
- tokenize(): normalized text split into words
- condition_sort_key(): order conditions as clean, then noisy from mild to heavy
"""
import re


def normalize_text(text) -> str:
    if text is None or (isinstance(text, float) and text != text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text) -> list:
    return normalize_text(text).split()


def condition_sort_key(cond: str):
    if cond == "clean":
        return (0, 0)
    m = re.search(r"(\d+)dB", str(cond))
    if m:
        return (1, -int(m.group(1)))
    return (2, 0)
