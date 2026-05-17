# ArXiv AI Analyzer

An AI-powered pipeline that fetches papers from ArXiv, analyzes them using multi-agent debate, and generates a styled PDF report.

## Overview

| Feature | Description |
|---------|-------------|
| **Input** | ArXiv CS papers (by category, date range, or CSV) |
| **Processing** | 3-stage pipeline: abstract analysis → debates → verdict synthesis |
| **Output** | PDF report with markdown intermediate |
| **Backend** | vLLM server (OpenAI-compatible API) |

## Quick Start

```bash
# Fetch today's papers from all CS categories → analyze → generate report.pdf
python main.py --base-url http://192.168.170.76:8000/v1

# Fetch last 3 days, only AI + CL categories, limit to 20 papers
python main.py --days 3 --categories AI,CL --limit 20 --base-url http://192.168.170.76:8000/v1

# Use an existing CSV (skip fetching)
python main.py --csv test_papers.csv --base-url http://192.168.170.76:8000/v1
```

## Output Structure

```
arxiv_run_20260421_223000/          # timestamped output dir
├── report.pdf                      # THE FINAL REPORT (WeasyPrint, A4, styled)
├── report.md                       # markdown intermediate
├── stage_a.jsonl                   # all 5-agent abstract analyses
├── stage_b.jsonl                   # all debate transcripts + judge verdicts
├── pdfs/                           # downloaded paper PDFs
│   ├── 2604.18578v1.pdf
│   └── ...
├── papers.csv                      # fetched papers (if not using --csv)
└── meta.json                       # run stats (timing, counts)
```

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--csv FILE` | — | Skip fetching, use existing CSV |
| `--days N` | `1` | How many past days to fetch |
| `--categories AI,CL,CV` | `all` | Which CS categories to fetch |
| `--max-per-cat N` | `5000` | Max results per category |
| `--limit N` | `all` | Cap papers processed (for testing) |
| `--base-url URL` | `localhost:8000/v1` | vLLM server address |
| `--output-dir DIR` | auto-timestamped | Where to write output |
| `--stage-a-workers N` | `50` | Concurrency for abstract analysis |
| `--stage-b-workers N` | `20` | Concurrency for debates |
| `--download-rate-limit N` | `3.0` | Seconds between PDF downloads |
| `-v`, `--verbose` | off | Enable debug logging |

## Requirements

- Python 3.10+
- vLLM server running (OpenAI-compatible)
- WeasyPrint (for PDF generation)
- See `requirements.txt` for full dependencies
