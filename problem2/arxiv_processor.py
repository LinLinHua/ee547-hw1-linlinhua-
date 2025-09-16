#!/usr/bin/env python3
# arxiv_processor.py — Part A (API/parse), Part B (per-abstract stats),
# Part C (outputs), Part D (robust error handling). stdlib-only.

# -------------------- Imports --------------------
import sys  # CLI args
import os  # filesystem
import json  # JSON I/O
import time  # timing / sleep
import datetime  # UTC timestamps
import re  # regex
import xml.etree.ElementTree as ET  # XML parsing
from urllib import request, parse, error  # HTTP + errors

# -------------------- Constants (Part A) --------------------
ARXIV_API = "http://export.arxiv.org/api/query"  # Atom API endpoint
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}  # namespaces

# -------------------- Tokenization / Terms (Part B) --------------------
# WORD_RE counts ASCII letters/digits and also Unicode letters (no digits/underscore)
WORD_RE = re.compile(r"(?:[A-Za-z0-9]+|[^\W\d_]+)", re.UNICODE)  # case-insensitive counting done via .lower()
SENT_SPLIT_RE = re.compile(r"[.!?]+")           # sentence split
UPPER_TERM_RE = re.compile(r"\b(?=\w*[A-Z])\w+\b")  # has ASCII uppercase
NUMERIC_TERM_RE = re.compile(r"\b\w*\d\w*\b")       # has digits
HYPHEN_TERM_RE = re.compile(r"\b\w+(?:-\w+)+\b")    # hyphenated terms

# -------------------- Helpers (A/C/D) --------------------
def utc_now_iso():
    """UTC ISO-8601 with 'Z'."""
    return datetime.datetime.utcnow().isoformat() + "Z"

def ensure_outdir(path):
    """Create output dir if missing."""
    os.makedirs(path, exist_ok=True)

def clean_text(s: str) -> str:
    """Collapse whitespace."""
    return " ".join((s or "").split())

def extract_text(elem, path: str) -> str:
    """findtext with ns + strip."""
    return (elem.findtext(path, default="", namespaces=ATOM_NS) or "").strip()

def build_query_url(search_query: str, max_results: int, start: int = 0) -> str:
    """Build API URL."""
    qs = parse.urlencode({"search_query": search_query, "start": start, "max_results": max_results})
    return f"{ARXIV_API}?{qs}"

def fetch_bytes(url: str, timeout: int = 10) -> bytes:
    """HTTP GET with urllib; return response bytes."""
    req = request.Request(url, headers={"User-Agent": "arxiv-processor/1.0 (+urllib.request)"})
    with request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def split_sentences(text: str):
    """Split on . ! ? and trim."""
    return [s.strip() for s in SENT_SPLIT_RE.split(text or "") if s.strip()]

def log_line(ts: str, msg: str) -> str:
    """Format log line."""
    return f"{ts} {msg}"

# -------------------- Part A: XML parsing --------------------
def parse_arxiv_xml(xml_bytes: bytes):
    """Yield paper dicts (raise ET.ParseError if root invalid)."""
    root = ET.fromstring(xml_bytes)  # may raise ParseError
    for entry in root.findall("atom:entry", ATOM_NS):
        id_url = extract_text(entry, "atom:id")
        arxiv_id = id_url.rsplit("/", 1)[-1] if id_url else ""

        title = clean_text(extract_text(entry, "atom:title"))
        abstract = clean_text(extract_text(entry, "atom:summary"))

        authors = []
        for a in entry.findall("atom:author", ATOM_NS):
            name = extract_text(a, "atom:name")
            if name:
                authors.append(name)

        categories = []
        for c in entry.findall("atom:category", ATOM_NS):
            term = c.attrib.get("term", "")
            if term:
                categories.append(term)
        primary = entry.find("arxiv:primary_category", ATOM_NS)
        if primary is not None:
            term = primary.attrib.get("term", "")
            if term and term not in categories:
                categories.append(term)

        published = extract_text(entry, "atom:published")
        updated = extract_text(entry, "atom:updated")

        yield {
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "categories": categories,
            "published": published,
            "updated": updated,
        }

# -------------------- Part B: per-abstract stats --------------------
def abstract_stats(abstract: str) -> dict:
    """Compute counts/means per abstract (case-insensitive counting)."""
    tokens = WORD_RE.findall(abstract or "")
    total_words = len(tokens)
    unique_words = len(set(t.lower() for t in tokens))
    sentences = split_sentences(abstract)
    total_sentences = len(sentences)
    wps = [len(WORD_RE.findall(s)) for s in sentences] if sentences else []
    avg_words_per_sentence = (sum(wps) / total_sentences) if total_sentences else 0.0
    avg_word_length = (sum(len(t) for t in tokens) / total_words) if total_words else 0.0
    return {
        "total_words": total_words,
        "unique_words": unique_words,
        "total_sentences": total_sentences,
        "avg_words_per_sentence": avg_words_per_sentence,
        "avg_word_length": avg_word_length,
    }

# -------------------- Part C: corpus aggregate --------------------
def corpus_aggregate(papers: list) -> dict:
    """Aggregate across abstracts (case-insensitive frequencies)."""
    total_words_all = 0
    global_vocab = set()
    lengths = []
    freq = {}
    doc_freq = {}
    upper_terms = set()
    numeric_terms = set()
    hyphen_terms = set()
    cat_dist = {}

    for p in papers:
        abs_text = p["abstract"]
        tokens = WORD_RE.findall(abs_text or "")
        tokens_lc = [t.lower() for t in tokens]

        total_words_all += len(tokens)
        lengths.append(len(tokens))
        global_vocab.update(tokens_lc)

        for w in tokens_lc:
            freq[w] = freq.get(w, 0) + 1
        for w in set(tokens_lc):
            doc_freq[w] = doc_freq.get(w, 0) + 1

        upper_terms.update(UPPER_TERM_RE.findall(abs_text or ""))
        numeric_terms.update(NUMERIC_TERM_RE.findall(abs_text or ""))
        hyphen_terms.update(HYPHEN_TERM_RE.findall(abs_text or ""))

        for c in p["categories"]:
            cat_dist[c] = cat_dist.get(c, 0) + 1

    top_50 = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:50]
    top_50_objs = [{"word": w, "frequency": f, "documents": doc_freq.get(w, 0)} for (w, f) in top_50]

    total_abs = len(papers)
    avg_len = (sum(lengths) / total_abs) if total_abs else 0.0
    longest = max(lengths) if lengths else 0
    shortest = min(lengths) if lengths else 0

    return {
        "corpus_stats": {
            "total_abstracts": total_abs,
            "total_words": total_words_all,
            "unique_words_global": len(global_vocab),
            "avg_abstract_length": avg_len,
            "longest_abstract_words": longest,
            "shortest_abstract_words": shortest,
        },
        "top_50_words": top_50_objs,
        "technical_terms": {
            "uppercase_terms": sorted(upper_terms),
            "numeric_terms": sorted(numeric_terms),
            "hyphenated_terms": sorted(hyphen_terms),
        },
        "category_distribution": cat_dist,
    }

# -------------------- Part D: validation / missing fields --------------------
def validate_required(p: dict):
    """Return (ok, missing_fields_set)."""
    required = {"arxiv_id", "title", "abstract", "published", "updated"}
    missing = {k for k in required if not p.get(k)}
    return (len(missing) == 0), missing

# -------------------- Main (A+B+C+D) --------------------
def main():
    """CLI: <search_query> <max_results 1..100> <output_dir>."""
    if len(sys.argv) != 4:
        print("Usage: python arxiv_processor.py <search_query> <max_results 1..100> <output_dir>", file=sys.stderr)
        sys.exit(1)

    query = sys.argv[1]
    try:
        max_results = int(sys.argv[2])
    except ValueError:
        print("max_results must be an integer between 1 and 100.", file=sys.stderr)
        sys.exit(2)
    if not (1 <= max_results <= 100):
        print("max_results must be between 1 and 100.", file=sys.stderr)
        sys.exit(3)
    out_dir = sys.argv[3]
    ensure_outdir(out_dir)

    # Output paths (Part C)
    papers_path = os.path.join(out_dir, "papers.json")
    corpus_path = os.path.join(out_dir, "corpus_analysis.json")
    log_path = os.path.join(out_dir, "processing.log")

    # Logging + timing (Part C)
    t0 = time.perf_counter()
    logs = []
    logs.append(log_line(utc_now_iso(), f"Starting ArXiv query: {query}"))

    # Build URL (Part A)
    url = build_query_url(query, max_results, start=0)

    # ----- Part D: network + 429 retry -----
    xml_bytes = b""
    attempts = 0
    max_attempts = 3
    while attempts < max_attempts:
        try:
            xml_bytes = fetch_bytes(url, timeout=10)  # HTTP GET
            break  # success
        except error.HTTPError as e:
            if e.code == 429 and attempts < max_attempts - 1:
                logs.append(log_line(utc_now_iso(), f"HTTP 429 (rate limited), retry in 3s (attempt {attempts+2}/{max_attempts})"))
                time.sleep(3)
                attempts += 1
                continue
            # non-429 HTTP error → network failure for our purposes
            logs.append(log_line(utc_now_iso(), f"HTTPError {e.code}: {e.reason} for URL: {url}"))
            # Write log and exit code 1 (network error)
            with open(log_path, "w", encoding="utf-8") as f:
                for line in logs:
                    f.write(line + "\n")
            sys.exit(1)
        except error.URLError as e:
            logs.append(log_line(utc_now_iso(), f"URLError: {e.reason} for URL: {url}"))
            with open(log_path, "w", encoding="utf-8") as f:
                for line in logs:
                    f.write(line + "\n")
            sys.exit(1)
        except Exception as e:
            logs.append(log_line(utc_now_iso(), f"Exception: {e} for URL: {url}"))
            with open(log_path, "w", encoding="utf-8") as f:
                for line in logs:
                    f.write(line + "\n")
            sys.exit(1)

    # If retries exhausted with no data (defensive; should have exited already)
    if not xml_bytes:
        logs.append(log_line(utc_now_iso(), "No data received after retries"))
        with open(log_path, "w", encoding="utf-8") as f:
            for line in logs:
                f.write(line + "\n")
        sys.exit(1)

    # ----- Part D: invalid XML handling -----
    raw_entries = []
    try:
        for rec in parse_arxiv_xml(xml_bytes):
            raw_entries.append(rec)
    except ET.ParseError as e:
        # Malformed XML: log error, continue with zero entries
        logs.append(log_line(utc_now_iso(), f"XMLParseError (root): {e}"))
        raw_entries = []

    logs.append(log_line(utc_now_iso(), f"Fetched {len(raw_entries)} results from ArXiv API"))

    # Process entries (Part B + C + D)
    papers = []
    for rec in raw_entries:
        arxiv_id = rec.get("arxiv_id", "")
        logs.append(log_line(utc_now_iso(), f"Processing paper: {arxiv_id if arxiv_id else '(unknown)'}"))

        ok, missing = validate_required(rec)
        if not ok:
            # Missing required fields: skip and warn (Part D)
            miss_str = ",".join(sorted(missing))
            logs.append(log_line(utc_now_iso(), f"WARNING missing fields for paper {arxiv_id or '(unknown)'}: {miss_str} (skipped)"))
            continue

        stats = abstract_stats(rec["abstract"])  # per-abstract stats (Part B)

        # Part C: papers.json item
        papers.append({
            "arxiv_id": rec["arxiv_id"],
            "title": rec["title"],
            "authors": rec["authors"],
            "abstract": rec["abstract"],
            "categories": rec["categories"],
            "published": rec["published"],
            "updated": rec["updated"],
            "abstract_stats": stats,
        })

    # Part C: corpus_analysis.json
    agg = corpus_aggregate(papers)
    corpus_payload = {
        "query": query,
        "papers_processed": len(papers),
        "processing_timestamp": utc_now_iso(),
        **agg,
    }

    # Write outputs (Part C)
    with open(papers_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus_payload, f, ensure_ascii=False, indent=2)

    # Final log line (Part C)
    elapsed = time.perf_counter() - t0
    logs.append(log_line(utc_now_iso(), f"Completed processing: {len(papers)} papers in {elapsed:.2f} seconds"))

    with open(log_path, "w", encoding="utf-8") as f:
        for line in logs:
            f.write(line + "\n")

# -------------------- Entrypoint --------------------
if __name__ == "__main__":
    main()
