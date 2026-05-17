"""
APPROACH C — Tool-Use Orchestrator Pipeline

A SINGLE agent with tools it can call to analyze papers.
The agent decides WHICH analysis steps to take per paper —
it's not a fixed pipeline, it's an agentic loop.
"""

import os
import re
import sys
import json
import time
from pathlib import Path

# Ensure we can import from the repo root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ai import (
    AIAgent, AIConfig, Chat,
    ToolCall, ToolResult, DoneEvent, Text, AgentResult, StepResult,
)
from load_papers import load_papers

# ── Configuration ──────────────────────────────────────────────────────────────

CONFIG = AIConfig(base_url="http://192.168.170.76:8000/v1", api_key="EMPTY")
MODEL = "/home/ng6309/datascience/santhosh/models/qwen3-6-27b"
OUTPUT_DIR = ROOT / "approach_c_run"

# ── Tool Functions ─────────────────────────────────────────────────────────────
# Each tool is a Python function with a docstring. AIAgent auto-converts them
# to tool schemas via fn_to_tool. Each tool internally creates a sub-agent call.

def extract_fields(title: str, abstract: str) -> str:
    """Extract structured fields from a research paper: problem statement, method,
    contribution, key results, datasets used, and baselines compared against.

    Args:
        title: The paper title.
        abstract: The paper abstract.
    """
    agent = AIAgent(config=CONFIG)
    prompt = (
        f"Extract the following structured fields from this research paper.\n"
        f"Return each field on its own line in the format 'FIELD: value'.\n\n"
        f"Title: {title}\n"
        f"Abstract: {abstract}\n\n"
        f"Fields to extract:\n"
        f"  - PROBLEM: What problem does the paper address?\n"
        f"  - METHOD: What is the core method or approach?\n"
        f"  - CONTRIBUTION: What is the main claim of contribution?\n"
        f"  - KEY_RESULTS: What are the main quantitative/qualitative results?\n"
        f"  - DATASETS: What datasets or benchmarks are used?\n"
        f"  - BASELINES: What prior methods are compared against?"
    )
    chat = agent.task(prompt, mode="instruct_reasoning", model=MODEL)
    return chat.answer


def assess_novelty(title: str, abstract: str) -> str:
    """Assess the novelty of a research paper's contribution relative to prior work.

    Args:
        title: The paper title.
        abstract: The paper abstract.
    """
    agent = AIAgent(config=CONFIG)
    prompt = (
        f"Assess the NOVELTY of this research paper. Be specific and critical.\n\n"
        f"Title: {title}\n"
        f"Abstract: {abstract}\n\n"
        f"Address:\n"
        f"  1. What is genuinely new vs. incremental?\n"
        f"  2. Does it build on well-known ideas in a new way?\n"
        f"  3. Is the insight surprising or expected?\n"
        f"  4. Rate novelty on a scale of 1-10 with justification."
    )
    chat = agent.task(prompt, mode="instruct_reasoning", model=MODEL)
    return chat.answer


def attack_claims(title: str, abstract: str) -> str:
    """Perform a skeptical attack on a research paper's claims, identifying weaknesses,
    potential issues, and alternative explanations.

    Args:
        title: The paper title.
        abstract: The paper abstract.
    """
    agent = AIAgent(config=CONFIG)
    prompt = (
        f"You are a SKEPTICAL reviewer. Critically attack the claims in this paper.\n\n"
        f"Title: {title}\n"
        f"Abstract: {abstract}\n\n"
        f"Address:\n"
        f"  1. What are the weakest claims or overstatements?\n"
        f"  2. What experiments or evidence are missing?\n"
        f"  3. What are plausible alternative explanations?\n"
        f"  4. Are the baselines fair and comprehensive?\n"
        f"  5. Could results be due to confounding factors?\n"
        f"  6. Rate claim strength on a scale of 1-10 with justification."
    )
    chat = agent.task(prompt, mode="instruct_reasoning", model=MODEL)
    return chat.answer


def tag_topics(title: str, abstract: str) -> str:
    """Tag a research paper with relevant topics, subfields, and keywords.

    Args:
        title: The paper title.
        abstract: The paper abstract.
    """
    agent = AIAgent(config=CONFIG)
    prompt = (
        f"Tag this research paper with relevant topics and keywords.\n\n"
        f"Title: {title}\n"
        f"Abstract: {abstract}\n\n"
        f"Return:\n"
        f"  - PRIMARY_FIELD: The main research area (e.g., NLP, CV, RL, Systems)\n"
        f"  - TOPICS: A comma-separated list of 5-8 specific topics\n"
        f"  - KEYWORDS: A comma-separated list of 8-12 keywords\n"
        f"  - APPLICATION_DOMAIN: Real-world domain if applicable"
    )
    chat = agent.task(prompt, mode="instruct_reasoning", model=MODEL)
    return chat.answer


def score_importance(title: str, abstract: str) -> str:
    """Score the overall importance and impact of a research paper.

    Args:
        title: The paper title.
        abstract: The paper abstract.
    """
    agent = AIAgent(config=CONFIG)
    prompt = (
        f"Score the overall IMPORTANCE and potential impact of this paper.\n\n"
        f"Title: {title}\n"
        f"Abstract: {abstract}\n\n"
        f"Address:\n"
        f"  1. How much could this change the field?\n"
        f"  2. Is it broadly applicable or narrow?\n"
        f"  3. Will it be widely cited?\n"
        f"  4. Rate on sub-scales (1-10): Technical Depth, Practical Impact, Breadth, Timeliness\n"
        f"  5. Overall importance score (1-10) with one-sentence justification."
    )
    chat = agent.task(prompt, mode="instruct_reasoning", model=MODEL)
    return chat.answer


# ── Orchestrator System Prompt ─────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM = """You are an AI research paper analyst. You have access to specialized tools
that each perform a different kind of analysis on a research paper.

For EVERY paper you receive, you MUST call ALL five analysis tools:
1. extract_fields — Extract structured fields (problem, method, contribution, results)
2. tag_topics — Tag with relevant topics and keywords
3. assess_novelty — Assess how novel the contribution is
4. attack_claims — Perform a skeptical critique of the claims
5. score_importance — Score the overall importance and impact

After collecting ALL tool results, synthesize them into a comprehensive final analysis.
Your final output must be a single markdown section containing:
- A brief summary of the paper
- The structured fields extracted
- Topic tags
- Novelty assessment
- Skeptical critique
- Importance scores
- A one-paragraph verdict

Do NOT skip any tool. Call all five, then produce the final analysis."""


def run_orchestrator(paper: dict) -> dict:
    """Run the orchestrator agent on a single paper using the forward() streaming API.
    Returns a result dict with all analysis data."""

    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    prompt = (
        f"Analyze this research paper using ALL your available tools.\n\n"
        f"**Title:** {title}\n\n"
        f"**Abstract:** {abstract}\n\n"
        f"Call every analysis tool, then produce a comprehensive final analysis."
    )

    agent = AIAgent(
        config=CONFIG,
        tools=[extract_fields, assess_novelty, attack_claims, tag_topics, score_importance],
        name="paper-orchestrator",
        description="Orchestrates paper analysis by calling specialized tool agents."
    )

    tool_calls_log = []
    steps_log = []
    final_answer = ""
    total_steps = 0
    total_tool_calls = 0
    total_elapsed = 0.0

    for event in agent.forward(
        Chat(ORCHESTRATOR_SYSTEM + "\n\n" + prompt),
        model=MODEL,
        mode="instruct_reasoning",
        tool_choice="auto",
        max_steps=15,
    ):
        if isinstance(event, ToolCall):
            tool_calls_log.append({
                "name": event.name,
                "arguments_preview": event.arguments[:120] if event.arguments else "",
            })
            print(f"  [tool] {event.name}")
        elif isinstance(event, ToolResult):
            print(f"  [result] {event.name}: {str(event.result)[:80]}...")
        elif isinstance(event, StepResult):
            steps_log.append({
                "step": event.step,
                "tool_calls": len(event.tool_calls),
                "tool_results": len(event.tool_results),
                "stop_reason": event.stop_reason,
            })
        elif isinstance(event, AgentResult):
            final_answer = event.answer
            total_steps = event.steps
            total_tool_calls = event.tool_calls_total
            total_elapsed = event.elapsed_s
            print(f"  [done] {total_steps} steps, {total_tool_calls} tool calls, {total_elapsed:.1f}s")
        elif isinstance(event, DoneEvent):
            break

    return {
        "arxiv_id": paper.get("arxiv_id", "unknown"),
        "title": title,
        "abstract": abstract,
        "tool_calls_made": tool_calls_log,
        "steps": steps_log,
        "total_steps": total_steps,
        "total_tool_calls": total_tool_calls,
        "elapsed_s": round(total_elapsed, 2),
        "final_analysis": final_answer,
    }


# ── Report Generation ──────────────────────────────────────────────────────────

def generate_report(results: list) -> str:
    """Generate a markdown report from results, sorted by importance score."""

    scored = []
    for r in results:
        analysis = r.get("final_analysis", "")
        # Try to find an overall score like "Overall: X/10" or "importance: X/10"
        m = re.search(r"(?:overall|importance).*(\d+(?:\.\d+)?)\s*/\s*10", analysis, re.IGNORECASE)
        if not m:
            m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", analysis)
        if m:
            score = float(m.group(1))
        else:
            score = 5.0  # default
        scored.append((score, r))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    lines = []
    lines.append("# ArXiv Paper Analysis Report\n")
    lines.append("**Approach:** Tool-Use Orchestrator (Approach C)\n")
    lines.append(f"**Papers analyzed:** {len(scored)}\n")
    lines.append(f"**Generated by:** approach_c.py\n\n")
    lines.append("---\n\n")
    lines.append("## Summary Table\n\n")
    lines.append("| Rank | Paper | ArXiv ID | Importance | Tool Calls | Time (s) |\n")
    lines.append("|------|-------|----------|------------|------------|----------|\n")

    for rank, (score, r) in enumerate(scored, 1):
        title_short = r["title"][:60] + "..." if len(r["title"]) > 60 else r["title"]
        lines.append(
            f"| {rank} | {title_short} | {r['arxiv_id']} | "
            f"{score:.1f}/10 | {r['total_tool_calls']} | {r['elapsed_s']}s |"
        )

    lines.append("\n---\n\n")
    lines.append("## Detailed Analyses\n\n")

    for rank, (score, r) in enumerate(scored, 1):
        lines.append(f"### {rank}. {r['title']} (Importance: {score:.1f}/10)\n\n")
        lines.append(f"- **ArXiv ID:** {r['arxiv_id']}\n")
        lines.append(f"- **Time:** {r['elapsed_s']}s\n")
        lines.append(f"- **Steps:** {r['total_steps']}\n")
        lines.append(f"- **Tool calls:** {r['total_tool_calls']}\n\n")

        lines.append("<details>\n<summary>Abstract</summary>\n\n")
        lines.append(f"{r['abstract']}\n\n")
        lines.append("</details>\n\n")

        lines.append("**Analysis:**\n\n")
        lines.append(f"{r['final_analysis']}\n\n")
        lines.append("---\n\n")

    return "\n".join(lines)


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("APPROACH C — Tool-Use Orchestrator Pipeline")
    print("=" * 70)

    # Load papers
    papers = load_papers()
    print(f"\nLoaded {len(papers)} papers from test_papers.csv")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    overall_start = time.monotonic()

    for i, paper in enumerate(papers, 1):
        pid = paper.get("arxiv_id", "unknown")
        t = paper.get("title", "Untitled")[:80]
        print(f"\n{'='*70}")
        print(f"Paper {i}/{len(papers)}: [{pid}] {t}...")
        print(f"{'='*70}")

        result = run_orchestrator(paper)
        results.append(result)

        elapsed_so_far = time.monotonic() - overall_start
        eta = (elapsed_so_far / i) * (len(papers) - i)
        print(f"  -> Completed in {result['elapsed_s']}s | ETA: {eta:.0f}s remaining")

    # Write results.jsonl
    jsonl_path = OUTPUT_DIR / "results.jsonl"
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(results)} results to {jsonl_path}")

    # Build report sorted by importance
    report_lines = generate_report(results)
    report_path = OUTPUT_DIR / "report.md"
    with open(report_path, "w") as f:
        f.write(report_lines)
    print(f"Wrote report to {report_path}")

    total_time = time.monotonic() - overall_start
    print(f"\n{'='*70}")
    print(f"TOTAL: {len(results)} papers processed in {total_time:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
