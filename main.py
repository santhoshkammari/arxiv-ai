#!/usr/bin/env python3
"""
ArXiv AI — Main CLI entry point.

Fetches today's papers, runs the 3-stage analysis pipeline, produces a report.

Usage:
  python main.py                          # Fetch last 1 day, all CS categories
  python main.py --days 3 --categories AI,CL
  python main.py --csv papers.csv         # Use existing CSV instead of fetching
  python main.py --limit 10               # Only process first 10 papers
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path
from datetime import datetime

from ai import AIConfig
from pipeline import ArxivPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("arxiv_main")


def fetch_papers(args) -> list[dict]:
    """Fetch papers from arXiv or load from CSV."""
    if args.csv:
        import pandas as pd
        logger.info(f"Loading papers from {args.csv}")
        df = pd.read_csv(args.csv)
        papers = df.to_dict("records")
        # Ensure required fields exist
        for p in papers:
            p.setdefault("arxiv_id", p.get("arxiv_id", "unknown"))
            p.setdefault("title", p.get("title", "Untitled"))
            p.setdefault("abstract", p.get("abstract", ""))
            p.setdefault("pdf_url", p.get("pdf_url", ""))
        logger.info(f"Loaded {len(papers)} papers from CSV")
        return papers

    from tool import fetch_latest_arxiv_cs_papers
    cats = [c.strip() for c in args.categories.split(",")] if args.categories else None
    logger.info(f"Fetching papers: days={args.days}, categories={cats or 'all'}")
    df = fetch_latest_arxiv_cs_papers(
        categories=cats,
        days=args.days,
        max_results_per_category=args.max_per_cat,
    )
    # Save CSV for reproducibility
    csv_path = Path(args.output_dir) / "papers.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"Fetched {len(df)} papers, saved to {csv_path}")
    return df.to_dict("records")


def main():
    parser = argparse.ArgumentParser(
        description="ArXiv AI — Daily research analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to existing CSV file (skip fetching)")
    parser.add_argument("--days", type=int, default=1,
                        help="Number of past days to fetch (default: 1)")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated category codes: AI,CL,CV,SE,IR,MA (default: all)")
    parser.add_argument("--max-per-cat", type=int, default=5000,
                        help="Max results per category (default: 5000)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of papers to process (for testing)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: arxiv_run_YYYYMMDD_HHMMSS)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="vLLM base URL (default: OPENAI_BASE_URL or http://localhost:8000/v1)")
    parser.add_argument("--stage-a-workers", type=int, default=50,
                        help="Max concurrent workers for Stage A (default: 50)")
    parser.add_argument("--stage-b-workers", type=int, default=20,
                        help="Max concurrent debate workers for Stage B (default: 20)")
    parser.add_argument("--download-rate-limit", type=float, default=3.0,
                        help="Seconds between PDF downloads (default: 3.0)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Set up output dir
    if args.output_dir is None:
        args.output_dir = f"arxiv_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Configure AI backend
    config = AIConfig()
    if args.base_url:
        config.base_url = args.base_url

    # Fetch papers
    papers = fetch_papers(args)

    if args.limit:
        papers = papers[:args.limit]
        logger.info(f"Limited to {len(papers)} papers")

    if not papers:
        logger.error("No papers to process!")
        sys.exit(1)

    # Run pipeline
    pipeline = ArxivPipeline(
        config=config,
        output_dir=args.output_dir,
        stage_a_workers=args.stage_a_workers,
        stage_b_workers=args.stage_b_workers,
        download_rate_limit=args.download_rate_limit,
    )

    report_path = pipeline.run(papers)

    print(f"\n{'='*60}")
    print(f"  ArXiv AI Report Generated!")
    print(f"  Report: {report_path}")
    print(f"  Output: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
