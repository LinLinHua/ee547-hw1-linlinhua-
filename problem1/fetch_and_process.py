"""""""""""""""
Your script must accept exactly two command line arguments:

1. Path to an input file containing URLs (one per line)
2. Path to output directory


For each URL in the input file, your script must:

1. Perform an HTTP GET request to the URL
2. Measure the response time in milliseconds
3. Capture the HTTP status code
4. Calculate the size of the response body in bytes
5. Count the number of words in the response (for text responses only)

Your script must write three files to the output directory:
1. responses.json  - per-URL details
2. summary.json    - aggregate statistics
3. errors.log      - one line per error
"""""""""""""""
#!/usr/bin/env python3

import sys  # for command-line arguments
import os   # for filesystem operations (paths, makedirs, file checks)
import json # for writing JSON outputs
import time # for precise timing (perf_counter)
import datetime  # for UTC timestamps
import re   # for regex-based word counting and charset extraction
from urllib import request, error  # stdlib HTTP client and error classes

# ---------- Helper functions ----------
def utc_now_iso():
    """Return current UTC time in ISO-8601 format with 'Z' suffix """
    # datetime.utcnow() gives naive UTC; we append 'Z' to make timezone explicit per spec
    return datetime.datetime.utcnow().isoformat() + "Z"

def ensure_outdir(path):
    """Create output directory if it doesn't exist."""
    # exist_ok=True means no error if the directory already exists
    os.makedirs(path, exist_ok=True)

def parse_urls(file_path):
    """Read a text file of URLs (one per line), strip whitespace, skip empty lines."""
    # Open the input file in UTF-8; read all lines; strip each; ignore blanks
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def extract_charset(content_type):
    """Extract 'charset=...' from a Content-Type header, return the charset or None if missing."""
    # If header is absent or empty, nothing to extract
    if not content_type:
        return None
    # Regex: find charset=VALUE (case-insensitive); capture the VALUE part
    m = re.search(r"charset=([\w\-\d_]+)", content_type, re.I)
    # If matched, return the first capture group; otherwise None
    return m.group(1) if m else None

def count_words_if_text(body_bytes, content_type):
    """
    For text responses, decode the body using the declared charset (or UTF-8 fallback) and count words.
    Word definition per spec: any sequence of alphanumeric characters [A-Za-z0-9]+.
    For non-text content types, return None.
    """
    # Only process when Content-Type explicitly indicates text
    if not content_type or "text" not in content_type.lower():
        return None
    # Try to detect charset from header; default to UTF-8 if missing
    charset = extract_charset(content_type) or "utf-8"
    try:
        # Decode using the declared charset; replace undecodable bytes to avoid failures
        text = body_bytes.decode(charset, errors="replace")
    except LookupError:
        # If the charset label is unknown to Python, fall back to UTF-8 with replacement
        text = body_bytes.decode("utf-8", errors="replace")
    # Count alphanumeric sequences; this excludes punctuation and whitespace
    return len(re.findall(r"[A-Za-z0-9]+", text))

def iso_log_line(ts_iso, url, msg):
    """Format a single error line: '[ISO-Z] [URL]: [message]'."""
    # Keep it compact and machine-readable
    return f"{ts_iso} {url}: {msg}"

# ---------- Per-URL processing ----------

def process_url(url):
    """
    Perform an HTTP GET for a single URL

    File 1: responses.json - Array of response data:

    [
      {
        "url": "[URL string]",
        "status_code": [integer],
        "response_time_ms": [float],
        "content_length": [integer],
        "word_count": [integer or null],
        "timestamp": "[ISO-8601 UTC]",
        "error": [null or error message string]
      },
      ...
    ]
    """
    # Measure response time in ms
    start = time.perf_counter()
    # record timestamp
    timestamp = utc_now_iso()

    # Build a Request with a basic User-Agent
    req = request.Request(url, method="GET", headers={
        "User-Agent": "fetch_and_process/1.0 (+stdlib urllib.request)"
    })

    try:
        # Open the URL with a 10-second timeout as required
        with request.urlopen(req, timeout=10) as resp:
            # Read entire response body as bytes
            body = resp.read()
            # Compute elapsed time in ms
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            # Status code: urllib adds .status
            status_code = getattr(resp, "status", None)
            # Content-Type header, used to decide if we should compute word count
            content_type = resp.headers.get("Content-Type", "")

            # Build the record for this URL (no error)
            return {
                "url": url,
                "status_code": status_code,
                "response_time_ms": elapsed_ms,
                "content_length": len(body),
                "word_count": count_words_if_text(body, content_type),  # None if not text/*
                "timestamp": timestamp,  # UTC ISO-8601 with 'Z'
                "error": None
            }

    except error.HTTPError as e:
        # HTTPError means the server responded with a non-2xx code
        # Read the body and headers to report content length and potential word count.
        try:
            body = e.read() or b""  # Some HTTP errors may still return a body
        except Exception:
            body = b""
        # Compute elapsed time even on failure
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        # Extract Content-Type if available to count words when text/*
        ctype = e.headers.get("Content-Type", "") if getattr(e, "headers", None) else ""
        # Return a record including the HTTP status and a descriptive error message
        return {
            "url": url,
            "status_code": e.code,  # HTTP status (e.g., 404)
            "response_time_ms": elapsed_ms,
            "content_length": len(body),
            "word_count": count_words_if_text(body, ctype),
            "timestamp": timestamp,
            "error": f"HTTPError {e.code}: {e.reason}"
        }

    except error.URLError as e:
        # URLError covers network-level failures (DNS, connection refused, timeout, etc.)
        # There is no HTTP status code because no response was received.
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "url": url,
            "status_code": None,     # No HTTP response
            "response_time_ms": elapsed_ms,
            "content_length": 0,     # Nothing downloaded
            "word_count": None,      # Not applicable
            "timestamp": timestamp,
            "error": f"URLError: {e.reason}"  # Include the reason string
        }

    except Exception as e:
        # Any unexpected exception: still measure time, record a message, and continue
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "url": url,
            "status_code": None,
            "response_time_ms": elapsed_ms,
            "content_length": 0,
            "word_count": None,
            "timestamp": timestamp,
            "error": f"Exception: {e}"
        }

# ---------- Aggregation ----------

def summarize(responses, started_iso, ended_iso):
    """
    File 2: summary.json - Aggregate statistics:

    {
      "total_urls": [integer],
      "successful_requests": [integer],
      "failed_requests": [integer],
      "average_response_time_ms": [float],
      "total_bytes_downloaded": [integer],
      "status_code_distribution": {
        "200": [count],
        "404": [count],
        ...
      },
      "processing_start": "[ISO-8601 UTC]",
      "processing_end": "[ISO-8601 UTC]"
    }
    """
    # Total number of URLs processed
    total = len(responses)
    # Successful requests are explicit 2xx with no error message
    successful = sum(
        1 for r in responses
        if r["error"] is None and isinstance(r["status_code"], int) and 200 <= r["status_code"] < 300
    )
    # Failures are everything else
    failed = total - successful
    # Average response time across all entries (include failures since we measured time)
    avg_rt = (sum(r.get("response_time_ms", 0.0) for r in responses) / total) if total else 0.0
    # Sum of all response body lengths
    total_bytes = sum(r.get("content_length", 0) for r in responses)

    # Build a histogram of status codes (only for integer codes)
    code_dist = {}
    for r in responses:
        sc = r.get("status_code")
        if isinstance(sc, int):
            key = str(sc)  # JSON keys should be strings
            code_dist[key] = code_dist.get(key, 0) + 1

    # Return the aggregate object as required by the spec
    return {
        "total_urls": total,
        "successful_requests": successful,
        "failed_requests": failed,
        "average_response_time_ms": avg_rt,
        "total_bytes_downloaded": total_bytes,
        "status_code_distribution": code_dist,
        "processing_start": started_iso,
        "processing_end": ended_iso,
    }

# ---------- Main ----------

def main():
    # Expect exactly two arguments: input URLs file path and output directory
    if len(sys.argv) != 3:
        # Print usage to stderr and exit
        print("Usage: python fetch_and_process.py <input_urls_file> <output_dir>", file=sys.stderr)
        sys.exit(1)

    # Unpack arguments
    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    # Validate that the input file exists; otherwise exit
    if not os.path.isfile(input_file):
        print(f"Input file not found: {input_file}", file=sys.stderr)
        sys.exit(2)

    # Ensure the output dir exists
    ensure_outdir(output_dir)

    # Parse URLs from the input file
    urls = parse_urls(input_file)

    # Record the processing start timestamp
    processing_start = utc_now_iso()

    # Prepare containers for per-URL results and error log lines
    responses = []
    error_lines = []

    # Iterate each URL and collect its result
    for url in urls:
        # Process one URL; always returns a well-formed record
        res = process_url(url)
        # Append to the results array
        responses.append(res)
        # If there was an error, append a formatted line to errors.log buffer
        if res["error"]:
            error_lines.append(iso_log_line(res["timestamp"], res["url"], res["error"]))

    # Record the processing end timestamp (UTC ISO 'Z')
    processing_end = utc_now_iso()

    # Build the aggregate summary from all responses
    summary = summarize(responses, processing_start, processing_end)

    # Compute output file paths inside the output directory
    responses_path = os.path.join(output_dir, "responses.json")
    summary_path = os.path.join(output_dir, "summary.json")
    errors_path = os.path.join(output_dir, "errors.log")

    # Write responses.json with pretty indentation; keep Unicode characters (ensure_ascii=False)
    with open(responses_path, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)

    # Write summary.json similarly
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Write errors.log: one line per error; if no errors, this file will be empty
    with open(errors_path, "w", encoding="utf-8") as f:
        for line in error_lines:
            f.write(line + "\n")

# Standard Python boilerplate to run main() when executed as a script
if __name__ == "__main__":
    main()
