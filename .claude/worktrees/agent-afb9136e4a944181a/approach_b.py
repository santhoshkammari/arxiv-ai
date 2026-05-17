"""
APPROACH B — Multi-Agent Debate Pipeline (Abstract-Only)

Differences from pipeline.py (Approach A):
  - NO PDF downloads; all analysis is abstract-only.
  - Stage A: 5 independent agents per paper, papers processed in parallel (max_workers=10).
  - Stage B: Judge agent per paper reads all 5 Stage A outputs and produces a verdict.
    Judges run in parallel (max_workers=10).
  - No Stage C (clustering/theme chapters) — report is a flat ranked list with debate highlights.

Uses:
  - AIAgent + AIConfig from ai.py
  - Pydantic models from models.py
  - load_papers() from load_papers.py (cached CSV, no arXiv fetching)
"""

import sys
import os
import json
import time
import logging
import threading
import concurrent.futures
from pathlib import Path
from datetime import datetime

from ai import AIAgent, AIConfig
from models import (
    ExtractionResult,
    NoveltyResult,
    SkepticResult,
    TopicTags,
    ReadGate,
    StageAResult,
    JudgeVerdict,
    StageBResult,
)

# ── Resolve repo root for load_papers import ──────────────────────────────
_REPO_ROOT = Path("/home/ntlpt24/Master/buildmode/personal/arxiv-ai")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from load_papers import load_papers

logger = logging.getLogger("approach_b")


# ── Constants ──────────────────────────────────────────────────────────────

MODEL_NAME = "/home/ng6309/datascience/santhosh/models/qwen3-6-27b"
MODEL_URL = "http://192.168.170.76:8000/v1"
MODE = "instruct_reasoning"
OUTPUT_DIR = Path(__file__).parent / "approach_b_run"
MAX_WORKERS_STAGE_A = 10
MAX_WORKERS_STAGE_B = 10


# ── Prompts (adapted from pipeline.py) ────────────────────────────────────

EXTRACTOR_PROMPT = """\
You are a precise research paper extractor. Given the title and abstract below, extract structured information.

Title: {title}
Abstract: {abstract}

Extract: the core problem, method/approach, main contribution, claimed results, datasets/benchmarks mentioned, and baseline methods compared against. Be specific and concise."""

NOVELTY_PROMPT = """\
You are a novelty assessor for AI/ML research. Given the title and abstract below, assess what is genuinely new.

Title: {title}
Abstract: {abstract}

Generate exactly 3 candidate "novel angles" — what is genuinely new here vs. reheated ideas. Rate each angle 1-5 for novelty. Provide an overall novelty score 1-5."""

SKEPTIC_PROMPT = """\
You are a research skeptic. Your job is to attack this paper's claims rigorously. Given the title and abstract below, find weaknesses.

Title: {title}
Abstract: {abstract}

Produce specific objections: unsubstantiated claims, ignored benchmarks, incremental contributions, methodological issues. Rate each attack's severity (low/medium/high) and give an overall credibility score 1-5."""

TOPIC_TAGGER_PROMPT = """\
You are a fine-grained topic tagger for AI/ML research. Given the title and abstract below, produce specific topic labels.

Title: {title}
Abstract: {abstract}

Generate fine-grained labels like: rlhf-reward-modeling, long-context-attention, vision-language-grounding, code-generation, multi-agent-debate, etc. NOT broad categories like "AI" or "ML". Be specific. Return 3-8 tags."""

READ_GATE_PROMPT = """\
You are a paper triage agent. Decide whether this paper is worth reading in full (downloading the PDF and doing deep analysis).

Title: {title}
Abstract: {abstract}

Consider: novelty, potential impact, methodological rigor, whether the claims are interesting enough to verify. Give a binary decision (worth_reading: true/false) with clear reasoning."""

JUDGE_PROMPT = """\
You are the Judge in a multi-agent paper analysis pipeline. You have received independent structured analyses from 5 specialist agents. Synthesize them into a verdict.

Paper Title: {title}
Paper Abstract: {abstract}

=== EXTRACTOR ===
Problem: {problem}
Method: {method}
Contribution: {contribution}
Claimed Results: {claimed_results}
Datasets: {datasets}
Baselines: {baselines}

=== NOVELTY ASSESSOR ===
Overall novelty: {novelty_score}/5
Summary: {novelty_summary}
Angles:
{novelty_angles}

=== SKEPTIC ===
Overall credibility: {credibility_score}/5
Attacks:
{skeptic_attacks}

=== TOPIC TAGGER ===
Tags: {tags}

=== READ GATE ===
Worth reading: {worth_reading}
Reasoning: {read_reasoning}

Based on all 5 agent analyses, produce your verdict:
- importance (1-10): How important is this paper for the field?
- confidence_in_claims (1-10): How confident should we be in the paper's claims, given the skeptic's attacks?
- standout_result: The single most noteworthy finding or result.
- open_questions: 2-4 open questions raised by the analysis.
- summary: A ~200-word summary incorporating insights from all 5 agents."""


# ── Stage A: 5 agents per paper, papers in parallel ───────────────────────

def _run_stage_a_agents(agent: AIAgent, paper: dict) -> StageAResult:
    """Run all 5 Stage A agents in parallel for a single paper."""
    title = paper["title"]
    abstract = paper["abstract"]

    def _call(prompt, schema):
        return agent.structured(prompt, schema, mode=MODE)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        f_extract = pool.submit(_call, EXTRACTOR_PROMPT.format(title=title, abstract=abstract), ExtractionResult)
        f_novelty = pool.submit(_call, NOVELTY_PROMPT.format(title=title, abstract=abstract), NoveltyResult)
        f_skeptic = pool.submit(_call, SKEPTIC_PROMPT.format(title=title, abstract=abstract), SkepticResult)
        f_topics = pool.submit(_call, TOPIC_TAGGER_PROMPT.format(title=title, abstract=abstract), TopicTags)
        f_gate = pool.submit(_call, READ_GATE_PROMPT.format(title=title, abstract=abstract), ReadGate)

        extraction = f_extract.result()
        novelty = f_novelty.result()
        skeptic = f_skeptic.result()
        topics = f_topics.result()
        read_gate = f_gate.result()

    # Coerce to proper Pydantic instances (structured() returns model or dict)
    extraction = extraction if isinstance(extraction, ExtractionResult) else ExtractionResult(**extraction)
    novelty = novelty if isinstance(novelty, NoveltyResult) else NoveltyResult(**novelty)
    skeptic = skeptic if isinstance(skeptic, SkepticResult) else SkepticResult(**skeptic)
    topics = topics if isinstance(topics, TopicTags) else TopicTags(**topics)
    read_gate = read_gate if isinstance(read_gate, ReadGate) else ReadGate(**read_gate)

    return StageAResult(
        arxiv_id=paper["arxiv_id"],
        title=title,
        extraction=extraction,
        novelty=novelty,
        skeptic=skeptic,
        topics=topics,
        read_gate=read_gate,
    )


def run_stage_a(agent: AIAgent, papers: list[dict], output_file: Path) -> list[StageAResult]:
    """Process all papers through Stage A agents in parallel."""
    results: list[StageAResult] = []
    lock = threading.Lock()

    # Truncate output file
    output_file.write_text("")

    def _process_and_save(paper: dict) -> StageAResult:
        try:
            result = _run_stage_a_agents(agent, paper)
        except Exception as e:
            logger.error(f"Stage A error for {paper['arxiv_id']}: {e}")
            raise
        with lock:
            output_file.open("a").write(result.model_dump_json() + "\n")
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_STAGE_A) as pool:
        futures = {pool.submit(_process_and_save, p): p for p in papers}
        for future in concurrent.futures.as_completed(futures):
            paper = futures[future]
            result = future.result()
            results.append(result)
            logger.info(
                f"Stage A | {paper['arxiv_id'][:16]} | "
                f"novelty={result.novelty.overall_novelty}/5 | "
                f"cred={result.skeptic.overall_credibility}/5 | "
                f"read_gate={'Y' if result.read_gate.worth_reading else 'N'}"
            )

    logger.info(f"Stage A complete: {len(results)}/{len(papers)} papers")
    return results


# ── Stage B: Judge per paper, all in parallel ─────────────────────────────

def _build_judge_prompt(paper: dict, stage_a: StageAResult) -> str:
    """Build the judge prompt from Stage A results."""
    ex = stage_a.extraction
    nv = stage_a.novelty
    sk = stage_a.skeptic
    tp = stage_a.topics
    rg = stage_a.read_gate

    novelty_angles = "\n".join(
        f"  - {a.angle} (rating: {a.rating}/5): {a.reasoning}"
        for a in nv.angles
    ) if nv.angles else "  (none)"

    skeptic_attacks = "\n".join(
        f"  - [{a.severity}] Claim: \"{a.claim}\" | Attack: {a.attack}"
        for a in sk.attacks
    ) if sk.attacks else "  (none)"

    return JUDGE_PROMPT.format(
        title=stage_a.title,
        abstract=paper["abstract"],
        problem=ex.problem,
        method=ex.method,
        contribution=ex.contribution,
        claimed_results=ex.claimed_results,
        datasets=", ".join(ex.datasets) or "None",
        baselines=", ".join(ex.baselines) or "None",
        novelty_score=nv.overall_novelty,
        novelty_summary=nv.summary,
        novelty_angles=novelty_angles,
        credibility_score=sk.overall_credibility,
        skeptic_attacks=skeptic_attacks,
        tags=", ".join(tp.tags),
        worth_reading=rg.worth_reading,
        read_reasoning=rg.reasoning,
    )


def _build_transcript(paper: dict, stage_a: StageAResult) -> str:
    """Build a full debate transcript from Stage A results for the StageBResult."""
    ex = stage_a.extraction
    nv = stage_a.novelty
    sk = stage_a.skeptic
    tp = stage_a.topics
    rg = stage_a.read_gate

    sections = []
    sections.append(f"[Extractor]:\n  Problem: {ex.problem}\n  Method: {ex.method}\n  Contribution: {ex.contribution}\n  Claimed Results: {ex.claimed_results}\n  Datasets: {', '.join(ex.datasets)}\n  Baselines: {', '.join(ex.baselines)}")
    sections.append(f"[Novelty Assessor]:\n  Overall: {nv.overall_novelty}/5\n  Summary: {nv.summary}" +
                    "".join(f"\n  Angle: {a.angle} ({a.rating}/5) — {a.reasoning}" for a in nv.angles))
    sections.append(f"[Skeptic]:\n  Credibility: {sk.overall_credibility}/5" +
                    "".join(f"\n  Attack [{a.severity}]: {a.attack} (vs claim: \"{a.claim}\")" for a in sk.attacks))
    sections.append(f"[Topic Tagger]: Tags: {', '.join(tp.tags)}")
    sections.append(f"[Read Gate]: Worth reading = {rg.worth_reading}. {rg.reasoning}")

    return "\n\n".join(sections)


def _run_judge(agent: AIAgent, paper: dict, stage_a: StageAResult) -> StageBResult:
    """Run the Judge agent for a single paper."""
    prompt = _build_judge_prompt(paper, stage_a)
    verdict = agent.structured(prompt, JudgeVerdict, mode=MODE)
    if not isinstance(verdict, JudgeVerdict):
        verdict = JudgeVerdict(**verdict)

    transcript = _build_transcript(paper, stage_a)

    return StageBResult(
        arxiv_id=paper["arxiv_id"],
        title=stage_a.title,
        round1_positions=[],  # populated from Stage A agents conceptually
        round2_rebuttals=[],
        verdict=verdict,
        full_transcript=transcript,
    )


def run_stage_b(
    agent: AIAgent,
    papers: list[dict],
    stage_a_results: list[StageAResult],
    output_file: Path,
) -> list[StageBResult]:
    """Process all papers through the Judge in parallel."""
    a_map = {r.arxiv_id: r for r in stage_a_results}
    results: list[StageBResult] = []
    lock = threading.Lock()

    # Truncate output file
    output_file.write_text("")

    def _judge_and_save(paper: dict) -> StageBResult:
        stage_a = a_map[paper["arxiv_id"]]
        try:
            result = _run_judge(agent, paper, stage_a)
        except Exception as e:
            logger.error(f"Stage B error for {paper['arxiv_id']}: {e}")
            raise
        with lock:
            output_file.open("a").write(result.model_dump_json() + "\n")
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_STAGE_B) as pool:
        futures = {pool.submit(_judge_and_save, p): p for p in papers}
        for future in concurrent.futures.as_completed(futures):
            paper = futures[future]
            result = future.result()
            results.append(result)
            logger.info(
                f"Stage B | {paper['arxiv_id'][:16]} | "
                f"importance={result.verdict.importance}/10 | "
                f"confidence={result.verdict.confidence_in_claims}/10"
            )

    logger.info(f"Stage B complete: {len(results)}/{len(papers)} papers judged")
    return results


# ── Report generation ─────────────────────────────────────────────────────

def generate_report(
    stage_a: list[StageAResult],
    stage_b: list[StageBResult],
    elapsed_s: float,
) -> str:
    """Generate report.md — papers sorted by judge importance with debate highlights."""
    a_map = {r.arxiv_id: r for r in stage_a}

    # Sort by importance descending
    ranked = sorted(stage_b, key=lambda r: r.verdict.importance, reverse=True)

    lines = []

    # Header
    lines.append("# ArXiv AI — Approach B Report (Multi-Agent Debate, Abstract-Only)")
    lines.append("")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"**Date:** {now}")
    lines.append(f"**Papers analyzed:** {len(ranked)}")
    lines.append(f"**Total time:** {elapsed_s:.1f}s")
    lines.append(f"**Approach:** 5-agent Stage A (Extractor, Novelty, Skeptic, Topic Tagger, Read Gate) "
                 f"+ Judge Stage B — all abstract-only, no PDF downloads")
    lines.append("")

    # Score table
    lines.append("## Score Summary")
    lines.append("")
    lines.append("| Rank | Paper | Importance | Confidence | Novelty | Credibility | Read Gate |")
    lines.append("|------|-------|-----------|------------|---------|-------------|-----------|")
    for i, b in enumerate(ranked, 1):
        a = a_map.get(b.arxiv_id)
        novelty = a.novelty.overall_novelty if a else "?"
        cred = a.skeptic.overall_credibility if a else "?"
        gate = "Yes" if (a and a.read_gate.worth_reading) else "No"
        title_short = b.title[:50] + ("..." if len(b.title) > 50 else "")
        lines.append(
            f"| {i} | [{title_short}](https://arxiv.org/abs/{b.arxiv_id}) "
            f"| {b.verdict.importance} | {b.verdict.confidence_in_claims} "
            f"| {novelty}/5 | {cred}/5 | {gate} |"
        )
    lines.append("")

    # Top 5 highlights
    lines.append("## Top 5 Papers (Debate Highlights)")
    lines.append("")
    for i, b in enumerate(ranked[:5], 1):
        a = a_map.get(b.arxiv_id)
        v = b.verdict
        lines.append(f"### {i}. {b.title}")
        lines.append(f"**ArXiv:** [`{b.arxiv_id}`](https://arxiv.org/abs/{b.arxiv_id})")
        lines.append("")
        lines.append(f"**Importance:** {v.importance}/10  "
                     f"**Confidence:** {v.confidence_in_claims}/10")
        lines.append("")

        if a:
            lines.append(f"**Problem:** {a.extraction.problem}")
            lines.append(f"**Method:** {a.extraction.method}")
            lines.append("")

            if a.novelty.angles:
                lines.append("**Novelty angles:**")
                for ang in a.novelty.angles:
                    lines.append(f"- {ang.angle} ({ang.rating}/5)")
                lines.append("")

            if a.skeptic.attacks:
                lines.append("**Skeptic attacks:**")
                for atk in a.skeptic.attacks:
                    lines.append(f"- [{atk.severity}] {atk.attack}")
                lines.append("")

            lines.append(f"**Topics:** {', '.join(a.topics.tags)}")
            lines.append("")
            lines.append(f"**Read Gate:** {'Yes' if a.read_gate.worth_reading else 'No'} — {a.read_gate.reasoning}")
            lines.append("")

        lines.append(f"**Judge summary:** {v.summary}")
        lines.append("")
        lines.append(f"**Standout result:** {v.standout_result}")
        lines.append("")
        if v.open_questions:
            lines.append("**Open questions:**")
            for q in v.open_questions:
                lines.append(f"- {q}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # All papers (compact)
    lines.append("## All Papers (Compact View)")
    lines.append("")
    for i, b in enumerate(ranked, 1):
        a = a_map.get(b.arxiv_id)
        v = b.verdict
        novelty = a.novelty.overall_novelty if a else "?"
        cred = a.skeptic.overall_credibility if a else "?"
        gate = "Y" if (a and a.read_gate.worth_reading) else "N"
        contribution = a.extraction.contribution[:120] if a else ""
        lines.append(
            f"{i}. **{b.title}** "
            f"(`{b.arxiv_id}`) — Imp:{v.importance} Conf:{v.confidence_in_claims} "
            f"Nov:{novelty}/5 Cred:{cred}/5 Gate:{gate} — {contribution}"
        )
    lines.append("")

    # Topic tag frequency
    lines.append("## Topic Tag Frequency")
    lines.append("")
    tag_counts: dict[str, int] = {}
    for a in stage_a:
        for t in a.topics.tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- `{tag}`: {count} paper(s)")
    lines.append("")

    # Methodology note
    lines.append("## Methodology: Approach B (Multi-Agent Debate)")
    lines.append("")
    lines.append("This report was generated using a multi-agent debate pipeline operating on abstracts only:")
    lines.append("")
    lines.append("**Stage A** — For each paper, 5 independent agents run in parallel:")
    lines.append("1. **Extractor** — Structured extraction of problem, method, contribution, results, datasets, baselines.")
    lines.append("2. **Novelty Assessor** — Identifies 3 novel angles, each rated 1-5, plus overall novelty score.")
    lines.append("3. **Skeptic** — Produces specific attacks on claims with severity ratings and overall credibility score.")
    lines.append("4. **Topic Tagger** — Assigns 3-8 fine-grained topic labels (not broad categories).")
    lines.append("5. **Read Gate** — Binary decision on whether the paper is worth reading in full, with reasoning.")
    lines.append("")
    lines.append("**Stage B** — A Judge agent reads all 5 Stage A outputs and produces a verdict:")
    lines.append("- Importance (1-10), Confidence in claims (1-10)")
    lines.append("- Standout result, Open questions, ~200-word summary")
    lines.append("")
    lines.append("Unlike a single-agent baseline that paraphrases the abstract, this approach forces:")
    lines.append("- **Divergence**: The Skeptic must find real weaknesses; Novelty must separate signal from hype.")
    lines.append("- **Structured synthesis**: The Judge integrates competing signals (e.g., high novelty but low credibility).")
    lines.append("- **Consistency**: Every paper gets the same 5 lenses applied in the same way.")
    lines.append("")
    lines.append(f"Processed {len(ranked)} papers in {elapsed_s:.1f}s using "
                 f"vLLM at {MODEL_URL} with model `{MODEL_NAME}`.")
    lines.append("")

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    t0 = time.monotonic()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    logger.info("=" * 70)
    logger.info("APPROACH B — Multi-Agent Debate Pipeline (Abstract-Only)")
    logger.info("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load papers from cached CSV — DO NOT fetch from arXiv
    papers = load_papers()
    logger.info(f"Loaded {len(papers)} papers from test_papers.csv")

    # Create AIAgent
    config = AIConfig(base_url=MODEL_URL)
    agent = AIAgent(config=config)

    # ── Stage A ────────────────────────────────────────────────────────
    logger.info("")
    logger.info(f"Stage A: {len(papers)} papers x 5 agents, max_workers={MAX_WORKERS_STAGE_A}")
    t_a = time.monotonic()
    stage_a_results = run_stage_a(agent, papers, OUTPUT_DIR / "stage_a.jsonl")
    t_a_elapsed = time.monotonic() - t_a
    logger.info(f"Stage A done in {t_a_elapsed:.1f}s ({t_a_elapsed / len(papers):.1f}s per paper)")

    # ── Stage B ────────────────────────────────────────────────────────
    logger.info("")
    logger.info(f"Stage B: Judge for {len(stage_a_results)} papers, max_workers={MAX_WORKERS_STAGE_B}")
    t_b = time.monotonic()
    stage_b_results = run_stage_b(agent, papers, stage_a_results, OUTPUT_DIR / "stage_b.jsonl")
    t_b_elapsed = time.monotonic() - t_b
    logger.info(f"Stage B done in {t_b_elapsed:.1f}s ({t_b_elapsed / len(papers):.1f}s per paper)")

    # ── Report ─────────────────────────────────────────────────────────
    total_elapsed = time.monotonic() - t0
    logger.info("")
    logger.info("Generating report...")

    report = generate_report(stage_a_results, stage_b_results, total_elapsed)
    report_path = OUTPUT_DIR / "report.md"
    report_path.write_text(report)
    logger.info(f"Report saved to {report_path}")

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Pipeline complete in {total_elapsed:.1f}s")
    logger.info(f"  Stage A: {t_a_elapsed:.1f}s ({len(stage_a_results)} papers)")
    logger.info(f"  Stage B: {t_b_elapsed:.1f}s ({len(stage_b_results)} papers)")
    logger.info(f"  Papers: {len(papers)}")
    importances = sorted([r.verdict.importance for r in stage_b_results], reverse=True)
    logger.info(f"  Importance scores: {importances}")
    logger.info(f"  Top paper: {stage_b_results[importances.index(max(importances))].title}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
