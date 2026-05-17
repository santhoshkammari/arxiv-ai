"""Cached paper loader — avoids refetching from arXiv on every run."""

import pandas as pd
from pathlib import Path

_DEFAULT_CSV = Path(__file__).parent / "test_papers.csv"


def load_papers(csv_path: str | Path = None, limit: int = None) -> list[dict]:
    """Load papers from a cached CSV file.

    Returns a list of dicts with keys: arxiv_id, title, abstract, pdf_url, etc.
    """
    path = Path(csv_path) if csv_path else _DEFAULT_CSV
    df = pd.read_csv(path)
    papers = df.to_dict("records")

    # Ensure required fields
    for p in papers:
        p.setdefault("arxiv_id", "unknown")
        p.setdefault("title", "Untitled")
        p.setdefault("abstract", "")
        p.setdefault("pdf_url", "")

    if limit:
        papers = papers[:limit]

    return papers
