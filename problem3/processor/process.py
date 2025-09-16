#!/usr/bin/env python3
# processor/process.py — Regex-based HTML text/link/image extractor and analyzer (stdlib only).

import os  # filesystem ops
import re  # regex extraction
import json  # JSON output
import time  # sleep + simple timing
from datetime import datetime  # UTC timestamp

# -------- Paths (fixed by spec) --------
STATUS_DIR = "/shared/status"  # status directory
RAW_DIR = "/shared/raw"  # input HTML directory
PROCESSED_DIR = "/shared/processed"  # output JSON directory
FETCH_DONE = os.path.join(STATUS_DIR, "fetch_complete.json")  # wait-for file
PROCESS_DONE = os.path.join(STATUS_DIR, "process_complete.json")  # final marker


# -------- Utilities --------
def utc_now_iso():
    """Return UTC ISO-8601 with trailing 'Z'."""
    return datetime.utcnow().isoformat() + "Z"


# -------- Required text extraction (use exactly this shape) --------
def strip_html(html_content):
    """Remove HTML tags and extract text."""
    # Remove script and style elements
    html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)

    # Extract links before removing tags
    links = re.findall(r'href=[\'"]?([^\'" >]+)', html_content, flags=re.IGNORECASE)

    # Extract images
    images = re.findall(r'src=[\'"]?([^\'" >]+)', html_content, flags=re.IGNORECASE)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', html_content)

    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text, links, images


# -------- Metrics helpers --------
WORD_RE = re.compile(r"(?:[A-Za-z0-9]+|[^\W\d_]+)", re.UNICODE)  # words (ASCII or Unicode letters/digits)
SENT_SPLIT_RE = re.compile(r"[.!?]+")


def count_words(text):
    """Count words (case-insensitive for counting)."""
    return len(WORD_RE.findall(text))


def avg_word_length(text):
    """Average word length (0.0 if no words)."""
    tokens = WORD_RE.findall(text)
    return (sum(len(t) for t in tokens) / len(tokens)) if tokens else 0.0


def count_sentences(text):
    """Count sentences split by . ! ?"""
    parts = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
    return len(parts)


def count_paragraphs_from_html(html_content, text_after_strip):
    """Count paragraph-like blocks via <p> tags; fallback to 1 if text exists else 0."""
    n = len(re.findall(r"<\s*p\b", html_content, flags=re.IGNORECASE))
    if n == 0:
        return 1 if text_after_strip else 0
    return n


# -------- Core processing --------
def process_one_file(src_path):
    """Read one HTML file, extract text/links/images, compute stats, return dict."""
    # Read file as UTF-8 with replacement to handle mixed encodings
    with open(src_path, "r", encoding="utf-8", errors="replace") as f:
        html = f.read()
    # Extract text/links/images using the required function
    text, links, images = strip_html(html)
    # Compute statistics
    stats = {
        "word_count": count_words(text),
        "sentence_count": count_sentences(text),
        "paragraph_count": count_paragraphs_from_html(html, text),
        "avg_word_length": avg_word_length(text),
    }
    # Build payload
    payload = {
        "source_file": os.path.basename(src_path),
        "text": text,
        "statistics": stats,
        "links": links,
        "images": images,
        "processed_at": utc_now_iso(),
    }
    return payload


def safe_write_json(path, obj):
    """Write JSON with UTF-8 and pretty indent."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    """Wait for fetch completion, process all HTML files, write outputs and status."""
    # Ensure required directories exist
    os.makedirs(STATUS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Wait for fetch_complete.json (polling)
    while not os.path.exists(FETCH_DONE):
        time.sleep(1)  # sleep 1s before re-check

    # List all HTML files in RAW_DIR (sorted for stable order)
    if not os.path.isdir(RAW_DIR):
        os.makedirs(RAW_DIR, exist_ok=True)  # ensure directory exists
    files = [os.path.join(RAW_DIR, n) for n in sorted(os.listdir(RAW_DIR)) if n.lower().endswith(".html")]

    processed_count = 0  # counter
    errors = []  # collect errors (optional debug)

    # Process each HTML file
    for src in files:
        try:
            payload = process_one_file(src)  # extract + analyze
            # Output file name mirrors input (e.g., page_N.html -> page_N.json)
            base = os.path.splitext(os.path.basename(src))[0]
            out_path = os.path.join(PROCESSED_DIR, f"{base}.json")
            safe_write_json(out_path, payload)  # save JSON
            processed_count += 1
        except Exception as e:
            # Keep going even if one file fails
            errors.append({"file": os.path.basename(src), "error": str(e), "time": utc_now_iso()})

    # Write process_complete.json with summary info
    completion = {
        "processed_files": processed_count,
        "errors": errors,  # optional diagnostics; can be empty
        "completed_at": utc_now_iso(),  # ISO-8601 UTC
        "status": "ok" if not errors else "completed_with_errors",
    }
    safe_write_json(PROCESS_DONE, completion)


if __name__ == "__main__":
    main()
