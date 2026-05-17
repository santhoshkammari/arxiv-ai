#!/usr/bin/env python3
"""
APPROACH A — Single-Agent Summary Pipeline

One agent, one structured LLM call per paper. No debate, no multi-round.
Processes 10 test papers in parallel and saves results to approach_a_run/.
"""

import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field

# Ensure the repo root is on sys.path so we can import ai.py and load_papers
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

# Also ensure the actual repo root (load_papers.py lives there, outside .claude/worktrees/)
_REPO_ROOT_ACTUAL = Path("/home/ntlpt24/Master/buildmode/personal/arxiv-ai")
if str(_REPO_ROOT_ACTUAL) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_ACTUAL))

from ai import AIAgent, AIConfig, Text, Assistant, DoneEvent
from load_papers import load_papers

# ── Pydantic schema for structured output ────────────────────────────────────

class PaperAnalysis(BaseModel):
    problem_statement: str = Field(description="The core problem the paper addresses, stated in your own words")
    method: str = Field(description="The main methodological approach")
    key_results: str = Field(description="Most important empirical or theoretical results with numbers")
    novelty_score: int = Field(ge=1, le=5, description="Novelty rating 1-5")
    novelty_reasoning: str = Field(description="Why this novelty score")
    critical_weaknesses: str = Field(description="Skeptic's view: real weaknesses, missing evaluations, etc.")
    topic_tags: list[str] = Field(description="3-8 fine-grained topic tags")
    summary: str = Field(description="Concise ~100-word summary for a busy researcher")
    importance_score: int = Field(ge=1, le=10, description="Importance score 1-10")


# ── Prompt template ──────────────────────────────────────────────────────────

PROMPT_FILE = REPO_ROOT / "approach_a" / "test_prompt.txt"
PROMPT_TEMPLATE = PROMPT_FILE.read_text()

OUTPUT_DIR = REPO_ROOT / "approach_a_run"


def make_prompt(paper: dict) -> str:
    """Fill the prompt template with paper data."""
    return PROMPT_TEMPLATE.format(
        title=paper.get("title", "Unknown"),
        arxiv_id=paper.get("arxiv_id", "unknown"),
        abstract=paper.get("abstract", ""),
    )


def analyze_paper(paper: dict, agent: AIAgent):
    """Run one structured analysis for a single paper. Returns (paper_id, result, elapsed)."""
    paper_id = paper.get("arxiv_id", "unknown")
    title = paper.get("title", "Unknown")
    prompt = make_prompt(paper)

    t0 = time.monotonic()
    result = agent.structured(prompt, schema=PaperAnalysis, mode="instruct_reasoning", max_tokens=4096)
    elapsed = round(time.monotonic() - t0, 2)

    return {
        "arxiv_id": paper_id,
        "title": title,
        "analysis": result,
        "elapsed_s": elapsed,
    }


def dictify(obj):
    """Recursively convert Pydantic models to dicts for JSON serialization."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: dictify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [dictify(v) for v in obj]
    return obj


def _get(a, key, default="N/A"):
    """Get value from dict or Pydantic model."""
    if isinstance(a, dict):
        return a.get(key, default)
    return getattr(a, key, default)


def generate_report(results: list[dict]) -> str:
    """Generate a markdown report sorted by importance_score descending."""
    def _score(r):
        a = r["analysis"]
        return _get(a, "importance_score", 0) or 0
    sorted_results = sorted(results, key=_score, reverse=True)

    lines = [
        "# ArXiv Daily — Approach A (Single-Agent Summary)",
        "",
        f"**{len(sorted_results)} papers analyzed** | Sorted by importance (high → low)",
        "",
        "---",
        "",
    ]

    for rank, r in enumerate(sorted_results, 1):
        a = r["analysis"]
        lines += [
            f"## {rank}. {r['title']}",
            "",
            f"- **ArXiv ID:** {r['arxiv_id']}",
            f"- **Importance:** {_get(a, 'importance_score', '?')}/10",
            f"- **Novelty:** {_get(a, 'novelty_score', '?')}/5",
            f"- **Tags:** {', '.join(_get(a, 'topic_tags', []))}",
            f"- **Processed in:** {r['elapsed_s']}s",
            "",
            "### Problem",
            "",
            f"{_get(a, 'problem_statement', 'N/A')}",
            "",
            "### Method",
            "",
            f"{_get(a, 'method', 'N/A')}",
            "",
            "### Key Results",
            "",
            f"{_get(a, 'key_results', 'N/A')}",
            "",
            "### Novelty Assessment",
            "",
            f"**Score:** {_get(a, 'novelty_score', '?')}/5",
            "",
            f"{_get(a, 'novelty_reasoning', 'N/A')}",
            "",
            "### Critical Weaknesses",
            "",
            f"{_get(a, 'critical_weaknesses', 'N/A')}",
            "",
            "### Summary",
            "",
            f"{_get(a, 'summary', 'N/A')}",
            "",
            "---",
            "",
        ]

    return "\n".join(lines)


def main():
    # ── Setup ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("APPROACH A — Single-Agent Summary Pipeline")
    print("=" * 60)

    # Load papers (from cached CSV, no arXiv fetching)
    papers = load_papers(REPO_ROOT / "test_papers.csv")
    print(f"\nLoaded {len(papers)} papers from test_papers.csv")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Create agent
    config = AIConfig(base_url="http://192.168.170.76:8000/v1")
    agent = AIAgent(config=config)
    model = "/home/ng6309/datascience/santhosh/models/qwen3-6-27b"

    # ── Process papers in parallel ───────────────────────────────────────────
    t_start = time.monotonic()
    results = []
    max_workers = min(10, len(papers))

    print(f"\nProcessing {len(papers)} papers in parallel (max_workers={max_workers})...")
    print("-" * 60)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_id = {
            pool.submit(analyze_paper, paper, agent): paper.get("arxiv_id", "unknown")
            for paper in papers
        }
        for future in as_completed(future_to_id):
            pid = future_to_id[future]
            try:
                result = future.result()
                results.append(result)
                imp = result["analysis"].get("importance_score", "?") if isinstance(result["analysis"], dict) else getattr(result["analysis"], "importance_score", "?")
                print(f"  [{len(results)}/{len(papers)}] {pid}: importance={imp}, time={result['elapsed_s']}s")
            except Exception as e:
                print(f"  [FAIL] {pid}: {e}")
                results.append({
                    "arxiv_id": pid,
                    "title": "Unknown",
                    "analysis": {"error": str(e)},
                    "elapsed_s": 0,
                })

    total_elapsed = round(time.monotonic() - t_start, 2)
    print(f"\nDone. {len(results)}/{len(papers)} papers processed in {total_elapsed}s total.")
    print(f"Average per paper: {total_elapsed / max(len(results), 1):.1f}s")

    # ── Save JSONL ───────────────────────────────────────────────────────────
    jsonl_path = OUTPUT_DIR / "stage_a.jsonl"
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(dictify(r), ensure_ascii=False) + "\n")
    print(f"\nSaved JSONL → {jsonl_path}")

    # ── Save report ──────────────────────────────────────────────────────────
    report_md = generate_report(results)
    report_path = OUTPUT_DIR / "report.md"
    report_path.write_text(report_md)
    print(f"Saved report → {report_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Papers processed: {len(results)}")
    print(f"Total time:       {total_elapsed}s")
    scores = []
    for r in results:
        a = r["analysis"]
        if isinstance(a, dict):
            s = a.get("importance_score")
        else:
            s = getattr(a, "importance_score", None)
        if s is not None:
            scores.append(s)
    if scores:
        print(f"Avg importance:   {sum(scores)/len(scores):.1f}/10")
        top = max(results, key=lambda r: (r["analysis"].get("importance_score", 0) if isinstance(r["analysis"], dict) else getattr(r["analysis"], "importance_score", 0)))
        print(f"Top paper:        {top['title']} (score={top['analysis'].get('importance_score') if isinstance(top['analysis'], dict) else top['analysis'].importance_score})")
    print(f"\nOutput files:")
    print(f"  {jsonl_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
