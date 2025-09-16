#!/usr/bin/env python3
# analyzer/analyze.py — Corpus-wide analysis on processed pages (stdlib only).

import os              # filesystem ops
import re              # tokenization and sentence split
import json            # JSON I/O
import time            # polling wait
from datetime import datetime  # UTC timestamp
from math import sqrt  # simple complexity proxy

# ----- Fixed paths -----
STATUS_DONE = "/shared/status/process_complete.json"   # wait-for file
PROCESSED_DIR = "/shared/processed"                    # input dir (page_*.json)
ANALYSIS_DIR = "/shared/analysis"                      # output dir
FINAL_REPORT = os.path.join(ANALYSIS_DIR, "final_report.json")  # output file

# ----- Tokenization / splitting -----
WORD_RE = re.compile(r"(?:[A-Za-z0-9]+|[^\W\d_]+)", re.UNICODE)  # words (ASCII or Unicode letters)
SENT_SPLIT_RE = re.compile(r"[.!?]+")  # sentence split on . ! ?

# ----- Time -----
def utc_now_iso():
    """Return UTC ISO-8601 with trailing 'Z'."""
    return datetime.utcnow().isoformat() + "Z"

# ----- Similarity (given) -----
def jaccard_similarity(doc1_words, doc2_words):
    """Calculate Jaccard similarity between two documents."""
    set1 = set(doc1_words)
    set2 = set(doc2_words)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0.0

# ----- Small helpers -----
def tokenize(text):
    """Return list of lowercased tokens (case-insensitive counting)."""
    return [t.lower() for t in WORD_RE.findall(text or "")]

def sentences(text):
    """Return list of sentence strings."""
    return [s.strip() for s in SENT_SPLIT_RE.split(text or "") if s.strip()]

def ngrams(tokens, n):
    """Yield n-grams (space-joined) from a token list."""
    L = len(tokens)
    for i in range(L - n + 1):
        yield " ".join(tokens[i:i+n])

# ----- Load all processed docs -----
def load_processed_docs():
    """Read all page_*.json from PROCESSED_DIR; return list of (name, text)."""
    docs = []
    if not os.path.isdir(PROCESSED_DIR):
        return docs
    for fn in sorted(os.listdir(PROCESSED_DIR)):
        if not fn.lower().endswith(".json"):
            continue
        path = os.path.join(PROCESSED_DIR, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            name = fn  # keep filename as id
            text = obj.get("text", "")  # fall back to empty if missing
            docs.append((name, text))
        except Exception:
            # Skip unreadable/invalid files silently (corpus should still process)
            continue
    return docs

# ----- Main analysis -----
def build_report(docs):
    """Compute global stats, similarity, n-grams, readability; return final dict."""
    # Accumulators
    total_words = 0
    vocab = set()
    word_counts = {}        # word -> count
    bigram_counts = {}      # "w1 w2" -> count
    trigram_counts = {}     # "w1 w2 w3" -> count
    total_sentences = 0
    total_word_len_sum = 0

    # Per-doc token bags (for Jaccard)
    doc_tokens = []         # list of (name, tokens_lower)

    # Process each document
    for name, text in docs:
        toks = tokenize(text)
        doc_tokens.append((name, toks))

        # Word totals
        total_words += len(toks)
        vocab.update(toks)
        for w in toks:
            word_counts[w] = word_counts.get(w, 0) + 1
            total_word_len_sum += len(w)

        # N-grams
        for bg in ngrams(toks, 2):
            bigram_counts[bg] = bigram_counts.get(bg, 0) + 1
        for tg in ngrams(toks, 3):
            trigram_counts[tg] = trigram_counts.get(tg, 0) + 1

        # Sentences
        sents = sentences(text)
        total_sentences += len(sents)

    # Top 100 words (count and relative frequency)
    top_words = sorted(word_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:100]
    top_100_words = [
        {"word": w, "count": c, "frequency": (c / total_words) if total_words else 0.0}
        for (w, c) in top_words
    ]

    # Document similarity (pairwise Jaccard)
    similarities = []
    N = len(doc_tokens)
    for i in range(N):
        name_i, toks_i = doc_tokens[i]
        for j in range(i + 1, N):
            name_j, toks_j = doc_tokens[j]
            sim = jaccard_similarity(toks_i, toks_j)
            similarities.append({"doc1": name_i, "doc2": name_j, "similarity": sim})

    # Top bigrams (show top 100 or fewer)
    top_bigrams = sorted(bigram_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:100]
    top_bigrams = [{"bigram": bg, "count": c} for (bg, c) in top_bigrams]

    # Top trigrams (show top 100 or fewer)
    top_trigrams = sorted(trigram_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:100]
    top_trigrams = [{"trigram": tg, "count": c} for (tg, c) in top_trigrams]


    # Readability metrics (corpus-level)
    avg_sentence_length = (total_words / total_sentences) if total_sentences else 0.0
    avg_word_length = (total_word_len_sum / total_words) if total_words else 0.0
    # Simple complexity proxy combining both scales
    complexity_score = sqrt(avg_sentence_length * avg_word_length) if (avg_sentence_length and avg_word_length) else 0.0

    # Assemble final report
    report = {
        "processing_timestamp": utc_now_iso(),
        "documents_processed": len(docs),
        "total_words": total_words,
        "unique_words": len(vocab),
        "top_100_words": top_100_words,
        "document_similarity": similarities,
        "top_bigrams": top_bigrams,
        "top_trigrams": top_trigrams,

        "readability": {
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
            "complexity_score": complexity_score
        }
    }
    return report

def main():
    """Wait for status, read processed files, compute analysis, write final report."""
    # Ensure output dir exists
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    # Wait for process_complete.json
    while not os.path.exists(STATUS_DONE):
        time.sleep(1)

    # Load processed documents
    docs = load_processed_docs()

    # Build report
    report = build_report(docs)

    # Save final report
    with open(FINAL_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
