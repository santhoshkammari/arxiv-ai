 # ArXiv AI Analyzer

An AI-powered tool that fetches papers from ArXiv, analyzes them using multi-agent debate, and generates a styled PDF report.

## Quick Start

Fetch today's papers from all CS categories → analyze → generate report.pdf
 python main.py --base-url http://192.168.170.76:8000/v1
 
 # Fetch last 3 days, only AI + CL categories, limit to 20 papers
 python main.py --days 3 --categories AI,CL --limit 20 --base-url http://192.168.170.76:8000/v1
 
 # Use an existing CSV (skip fetching)
 python main.py --csv test_papers.csv --base-url http://192.168.170.76:8000/v1

What it creates

 arxiv_run_20260421_223000/     ← timestamped output dir
 ├── report.pdf                 ← THE FINAL REPORT (WeasyPrint, A4, styled)
 ├── report.md                  ← markdown intermediate
 ├── stage_a.jsonl              ← all 5-agent abstract analyses
 ├── stage_b.jsonl              ← all debate transcripts + judge verdicts
 ├── pdfs/                      ← downloaded paper PDFs
 │   ├── 2604.18578v1.pdf
 │   └── ...
 ├── papers.csv                 ← fetched papers (if not using --csv)
 └── meta.json                  ← run stats (timing, counts)

All flags

┌─────────────────────────┬──────────────────┬────────────────────────────────────┐
│ Flag                    │ Default          │ What it does                       │
├─────────────────────────┼──────────────────┼────────────────────────────────────┤
│ --csv FILE              │ —                │ Skip fetching, use existing CSV    │
├─────────────────────────┼──────────────────┼────────────────────────────────────┤
│ --days N                │ 1                │ How many past days to fetch        │
├─────────────────────────┼──────────────────┼────────────────────────────────────┤
│ --categories AI,CL,CV   │ all              │ Which CS categories                │
├─────────────────────────┼──────────────────┼────────────────────────────────────┤
│ --limit N               │ all              │ Cap papers processed (for testing) │
├─────────────────────────┼──────────────────┼────────────────────────────────────┤
│ --base-url URL          │ localhost:8000   │ vLLM server address                │
├─────────────────────────┼──────────────────┼────────────────────────────────────┤
│ --output-dir DIR        │ auto-timestamped │ Where to write output              │
├─────────────────────────┼──────────────────┼────────────────────────────────────┤
│ --stage-a-workers N     │ 50               │ Concurrency for abstract analysis  │
├─────────────────────────┼──────────────────┼────────────────────────────────────┤
│ --stage-b-workers N     │ 20               │ Concurrency for debates            │
├─────────────────────────┼──────────────────┼────────────────────────────────────┤
│ -v                      │ off              │ Verbose debug logging              │
└─────────────────────────┴──────────────────┴────────────────────────────────────┘
