"""
ArXiv AI Pipeline — Stage A, B, C with concurrent execution.

Three-stage pipeline:
  A: Abstract Crunching (5 agents/paper, massively parallel)
  B: PDF Download + Deep Read (debate ensemble per paper)
  C: Cross-paper Synthesis (cluster → theme chapters → report)
"""

import os
import json
import time
import queue
import logging
import threading
import tempfile
import shutil
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import pymupdf4llm

from ai import AIAgent, AIConfig, Chat, Text, Assistant, DoneEvent, AgentResult
from models import (
    ExtractionResult, NoveltyResult, SkepticResult, TopicTags, ReadGate,
    StageAResult, DebatePosition, DebateRebuttal, JudgeVerdict, StageBResult,
    ClusterResult, ThemeChapter, ReportIntro,
)

logger = logging.getLogger("arxiv_pipeline")


# ── Prompts ─────────────────────────────────────────────────────────────────

EXTRACTOR_PROMPT = """You are a precise research paper extractor. Given the title and abstract below, extract structured information.

Title: {title}
Abstract: {abstract}

Extract: the core problem, method/approach, main contribution, claimed results, datasets/benchmarks mentioned, and baseline methods compared against. Be specific and concise."""

NOVELTY_PROMPT = """You are a novelty assessor for AI/ML research. Given the title and abstract below, assess what's genuinely new.

Title: {title}
Abstract: {abstract}

Generate exactly 3 candidate "novel angles" — what's genuinely new here vs. reheated. Rate each angle 1-5 for novelty. Provide an overall novelty score 1-5."""

SKEPTIC_PROMPT = """You are a research skeptic. Your job is to attack this paper's claims rigorously. Given the title and abstract below, find weaknesses.

Title: {title}
Abstract: {abstract}

Produce specific objections: unsubstantiated claims, ignored benchmarks, incremental contributions, methodological issues. Rate each attack's severity (low/medium/high) and give an overall credibility score 1-5."""

TOPIC_TAGGER_PROMPT = """You are a fine-grained topic tagger for AI/ML research. Given the title and abstract below, produce specific topic labels.

Title: {title}
Abstract: {abstract}

Generate fine-grained labels like: rlhf-reward-modeling, long-context-attention, vision-language-grounding, code-generation, multi-agent-debate, etc. NOT broad categories like "AI" or "ML". Be specific. Return 3-8 tags."""

READ_GATE_PROMPT = """You are a paper triage agent. Decide whether this paper is worth reading in full (downloading the PDF and doing deep analysis).

Title: {title}
Abstract: {abstract}

Consider: novelty, potential impact, methodological rigor, whether the claims are interesting enough to verify. Give a binary decision (worth_reading: true/false) with clear reasoning."""

ADVOCATE_PROMPT = """You are the Advocate. Make the strongest possible case for this paper. Why is it important? What would change if the claims are true?

Paper Title: {title}
Paper Content:
{content}

Previous debate context (Stage A analysis):
{stage_a_summary}

Write your advocacy position. Be specific, reference concrete claims and results."""

DEBATE_SKEPTIC_PROMPT = """You are the Skeptic in a paper debate. Attack the methodology, experimental design, baselines, and claims.

Paper Title: {title}
Paper Content:
{content}

Previous debate context (Stage A analysis):
{stage_a_summary}

Write specific, substantive objections. Not generic complaints — point to actual weaknesses."""

REPRODUCER_PROMPT = """You are the Reproducer. If you tried to implement this paper, what would you need? What's underspecified? What's the hidden cost?

Paper Title: {title}
Paper Content:
{content}

Analyze reproducibility: missing details, computational requirements, data availability, implementation complexity."""

CONTEXTUALIZER_PROMPT = """You are the Contextualizer. Given this paper and knowledge of other papers being analyzed today, find connections.

Paper Title: {title}
Paper Content:
{content}

Other papers analyzed today (summaries):
{other_papers_summary}

Find connections: papers claiming opposite things, papers using the same datasets, complementary approaches, building on each other."""

METHODOLOGIST_PROMPT = """You are the Methodologist. Focus only on the technical method. Explain it precisely.

Paper Title: {title}
Paper Content:
{content}

Explain the method clearly and precisely. What's the core technical innovation? How does it work step by step? What are the key equations/algorithms?"""

JUDGE_PROMPT = """You are the Judge. Read the full debate transcript below and produce a verdict.

Paper Title: {title}

== DEBATE TRANSCRIPT ==
{transcript}

Produce: importance (1-10), confidence in claims (1-10), the standout result, open questions, and a ~200-word summary incorporating the debate insights."""

ROUND2_PROMPT = """You are {agent_name}. You've read the other agents' positions. Write your rebuttal/update.

Paper Title: {title}

Your original position:
{own_position}

Other agents' positions:
{other_positions}

Update your analysis in light of the other perspectives. What do you agree with? Where do you still disagree? What new points emerge?"""

CLUSTERER_PROMPT = """You are a thematic clusterer. Given the topic tags from all analyzed papers, group them into coherent themes.

Papers and their tags:
{papers_tags}

Create 5-15 thematic clusters. Each cluster should have a descriptive name and include the arxiv IDs of papers that belong to it. Papers can belong to multiple clusters. Focus on creating meaningful, specific themes (not generic ones like "AI" or "ML")."""

THEME_WRITER_PROMPT = """You are a theme chapter writer for an AI research report. Write a comprehensive chapter about this theme.

Theme: {theme_name} — {theme_description}

Papers in this theme:
{papers_data}

Write a 1-2 page chapter covering:
1. What this theme is about
2. The 3-5 most important papers (use importance scores from Judge verdicts)
3. Cross-paper tensions (where Skeptics attacked, where papers disagree)
4. Open questions the theme raises

Use specific quotes from the debate transcripts where relevant. Write in engaging, analytical prose. Format as markdown."""

EDITOR_PROMPT = """You are the editor of a daily AI research report. Given the theme chapters below, write the introduction.

Date: {date}
Total papers analyzed: {total_papers}
Papers deeply reviewed: {deep_reviewed}

Theme chapters:
{theme_summaries}

Write:
1. 5 TL;DR bullets — "today's biggest stories"
2. "What to read first" — 5 picks with one-line pitches
3. Cross-theme observations — what patterns emerge across themes
4. A brief introduction setting the scene

Format as markdown. Be opinionated and insightful, not just descriptive."""


# ── Stage A: Abstract Crunching ─────────────────────────────────────────────

class StageA:
    """Fan-out: 5 agents per paper, all parallel via vLLM batching."""

    def __init__(self, agent: AIAgent, output_dir: Path, max_workers: int = 100):
        self.agent = agent
        self.output_dir = output_dir
        self.output_file = output_dir / "stage_a.jsonl"
        self.max_workers = max_workers
        self._lock = threading.Lock()

    def _run_single_agent(self, paper: dict, prompt_template: str, schema, **fmt_kwargs) -> dict:
        prompt = prompt_template.format(
            title=paper["title"],
            abstract=paper["abstract"],
            **fmt_kwargs,
        )
        try:
            result = self.agent.structured(prompt, schema, mode="instruct_reasoning")
            if hasattr(result, "model_dump"):
                return result.model_dump()
            return result
        except Exception as e:
            logger.error(f"Agent error for {paper.get('arxiv_id', '?')}: {e}")
            return {"error": str(e)}

    def _process_paper(self, paper: dict) -> StageAResult:
        """Run all 5 agents in parallel for a single paper."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            f_extract = pool.submit(self._run_single_agent, paper, EXTRACTOR_PROMPT, ExtractionResult)
            f_novelty = pool.submit(self._run_single_agent, paper, NOVELTY_PROMPT, NoveltyResult)
            f_skeptic = pool.submit(self._run_single_agent, paper, SKEPTIC_PROMPT, SkepticResult)
            f_topics = pool.submit(self._run_single_agent, paper, TOPIC_TAGGER_PROMPT, TopicTags)
            f_gate = pool.submit(self._run_single_agent, paper, READ_GATE_PROMPT, ReadGate)

            extraction = f_extract.result()
            novelty = f_novelty.result()
            skeptic = f_skeptic.result()
            topics = f_topics.result()
            gate = f_gate.result()

        result = StageAResult(
            arxiv_id=paper["arxiv_id"],
            title=paper["title"],
            extraction=extraction if isinstance(extraction, ExtractionResult) else ExtractionResult(**extraction),
            novelty=novelty if isinstance(novelty, NoveltyResult) else NoveltyResult(**novelty),
            skeptic=skeptic if isinstance(skeptic, SkepticResult) else SkepticResult(**skeptic),
            topics=topics if isinstance(topics, TopicTags) else TopicTags(**topics),
            read_gate=gate if isinstance(gate, ReadGate) else ReadGate(**gate),
        )
        return result

    def _append_result(self, result: StageAResult):
        with self._lock:
            with open(self.output_file, "a") as f:
                f.write(result.model_dump_json() + "\n")

    def run(self, papers: list[dict], read_queue: queue.Queue) -> list[StageAResult]:
        """Process all papers. Stream results to JSONL and push worth-reading to queue."""
        logger.info(f"Stage A: Processing {len(papers)} papers with {self.max_workers} workers")
        results = []

        def _process_and_queue(paper):
            result = self._process_paper(paper)
            self._append_result(result)
            if result.read_gate.worth_reading:
                read_queue.put({"paper": paper, "stage_a": result})
                logger.info(f"  → {paper['arxiv_id']} flagged for deep read")
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_process_and_queue, p): p for p in papers}
            for future in concurrent.futures.as_completed(futures):
                paper = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"  Stage A done: {paper['arxiv_id']} "
                                f"(novelty={result.novelty.overall_novelty}, "
                                f"read={result.read_gate.worth_reading})")
                except Exception as e:
                    logger.error(f"  Stage A FAILED for {paper.get('arxiv_id', '?')}: {e}")

        read_queue.put(None)  # sentinel
        logger.info(f"Stage A complete: {len(results)} papers processed, "
                     f"{sum(1 for r in results if r.read_gate.worth_reading)} flagged for deep read")
        return results


# ── Stage B: PDF Download + Debate ──────────────────────────────────────────

class StageB:
    """B.1: Download PDFs (rate-limited). B.2: Per-paper debate ensemble."""

    def __init__(self, agent: AIAgent, output_dir: Path,
                 pdf_dir: Path = None, max_debate_workers: int = 50,
                 download_rate_limit: float = 3.0):
        self.agent = agent
        self.output_dir = output_dir
        self.output_file = output_dir / "stage_b.jsonl"
        self.pdf_dir = pdf_dir or output_dir / "pdfs"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.max_debate_workers = max_debate_workers
        self.download_rate_limit = download_rate_limit
        self._lock = threading.Lock()
        self._all_stage_a_results: list = []

    def _download_pdf(self, paper: dict) -> Optional[str]:
        """Download PDF and extract markdown. Returns markdown text or None."""
        pdf_url = paper.get("pdf_url")
        if not pdf_url:
            return None

        arxiv_id = paper["arxiv_id"].replace("/", "_")
        pdf_path = self.pdf_dir / f"{arxiv_id}.pdf"

        if not pdf_path.exists():
            try:
                resp = requests.get(pdf_url, stream=True, timeout=60)
                resp.raise_for_status()
                with open(pdf_path, "wb") as f:
                    shutil.copyfileobj(resp.raw, f)
                logger.info(f"  Downloaded: {pdf_path.name}")
            except Exception as e:
                logger.error(f"  Download failed for {arxiv_id}: {e}")
                return None

        try:
            md_text = pymupdf4llm.to_markdown(str(pdf_path))
            # Truncate to avoid overwhelming the LLM
            if len(md_text) > 30000:
                md_text = md_text[:30000] + "\n\n[... truncated ...]"
            return md_text
        except Exception as e:
            logger.error(f"  PDF extraction failed for {arxiv_id}: {e}")
            return None

    def _format_stage_a_summary(self, stage_a: StageAResult) -> str:
        s = stage_a
        return (
            f"Extraction: {s.extraction.problem} | Method: {s.extraction.method}\n"
            f"Novelty: {s.novelty.overall_novelty}/5 — {s.novelty.summary}\n"
            f"Skeptic: credibility {s.skeptic.overall_credibility}/5, "
            f"attacks: {'; '.join(a.attack[:80] for a in s.skeptic.attacks[:3])}\n"
            f"Topics: {', '.join(s.topics.tags)}"
        )

    def _other_papers_summary(self, exclude_id: str) -> str:
        summaries = []
        for r in self._all_stage_a_results[:20]:  # cap at 20 for context
            if r.arxiv_id != exclude_id:
                summaries.append(
                    f"- {r.title}: {r.extraction.contribution} "
                    f"(novelty: {r.novelty.overall_novelty}/5)"
                )
        return "\n".join(summaries) if summaries else "No other papers available yet."

    def _run_debate(self, paper: dict, content: str, stage_a: StageAResult) -> StageBResult:
        """3-round debate ensemble for a single paper."""
        title = paper["title"]
        stage_a_summary = self._format_stage_a_summary(stage_a)
        other_papers = self._other_papers_summary(paper["arxiv_id"])

        # Round 1: Independent positions (5 parallel calls)
        agent_configs = [
            ("Advocate", ADVOCATE_PROMPT, {"content": content, "stage_a_summary": stage_a_summary}),
            ("Skeptic", DEBATE_SKEPTIC_PROMPT, {"content": content, "stage_a_summary": stage_a_summary}),
            ("Reproducer", REPRODUCER_PROMPT, {"content": content}),
            ("Contextualizer", CONTEXTUALIZER_PROMPT, {"content": content, "other_papers_summary": other_papers}),
            ("Methodologist", METHODOLOGIST_PROMPT, {"content": content}),
        ]

        def _get_position(name, prompt_tmpl, extra_kwargs):
            prompt = prompt_tmpl.format(title=title, **extra_kwargs)
            chat = self.agent.task(prompt, mode="instruct_reasoning")
            return DebatePosition(agent_name=name, position=chat.answer)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(_get_position, *cfg) for cfg in agent_configs]
            positions = [f.result() for f in futures]

        # Round 2: Rebuttals (5 parallel calls)
        def _get_rebuttal(position: DebatePosition):
            others_text = "\n\n".join(
                f"[{p.agent_name}]: {p.position}"
                for p in positions if p.agent_name != position.agent_name
            )
            prompt = ROUND2_PROMPT.format(
                agent_name=position.agent_name, title=title,
                own_position=position.position, other_positions=others_text,
            )
            chat = self.agent.task(prompt, mode="instruct_reasoning")
            return DebateRebuttal(
                agent_name=position.agent_name,
                rebuttal=chat.answer,
                key_disagreements=[],
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(_get_rebuttal, p) for p in positions]
            rebuttals = [f.result() for f in futures]

        # Build transcript
        transcript_parts = ["== ROUND 1: INDEPENDENT POSITIONS ==\n"]
        for p in positions:
            transcript_parts.append(f"[{p.agent_name}]:\n{p.position}\n")
        transcript_parts.append("\n== ROUND 2: REBUTTALS ==\n")
        for r in rebuttals:
            transcript_parts.append(f"[{r.agent_name}]:\n{r.rebuttal}\n")
        transcript = "\n".join(transcript_parts)

        # Round 3: Judge verdict
        judge_prompt = JUDGE_PROMPT.format(title=title, transcript=transcript)
        verdict = self.agent.structured(judge_prompt, JudgeVerdict, mode="instruct_reasoning")
        if not isinstance(verdict, JudgeVerdict):
            verdict = JudgeVerdict(**verdict)

        return StageBResult(
            arxiv_id=paper["arxiv_id"],
            title=title,
            round1_positions=[p.model_dump() if hasattr(p, 'model_dump') else p for p in positions],
            round2_rebuttals=[r.model_dump() if hasattr(r, 'model_dump') else r for r in rebuttals],
            verdict=verdict,
            full_transcript=transcript,
        )

    def _append_result(self, result: StageBResult):
        with self._lock:
            with open(self.output_file, "a") as f:
                f.write(result.model_dump_json() + "\n")

    def run(self, read_queue: queue.Queue, all_stage_a: list) -> list[StageBResult]:
        """Consume read_queue, download PDFs, run debates. Returns all verdicts."""
        self._all_stage_a_results = all_stage_a
        results = []
        debate_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_debate_workers)
        debate_futures = []

        def _download_and_debate(item):
            paper = item["paper"]
            stage_a = item["stage_a"]
            content = self._download_pdf(paper)
            if content is None:
                # Fall back to abstract-only debate
                content = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
            time.sleep(self.download_rate_limit)  # rate limit downloads
            result = self._run_debate(paper, content, stage_a)
            self._append_result(result)
            return result

        # Consume queue — downloads are sequential (rate limited), debates are parallel
        while True:
            item = read_queue.get()
            if item is None:
                break
            debate_futures.append(debate_pool.submit(_download_and_debate, item))

        # Collect results
        for future in concurrent.futures.as_completed(debate_futures):
            try:
                result = future.result()
                results.append(result)
                logger.info(f"  Stage B done: {result.arxiv_id} "
                            f"(importance={result.verdict.importance}/10)")
            except Exception as e:
                logger.error(f"  Stage B debate FAILED: {e}")

        debate_pool.shutdown(wait=True)
        logger.info(f"Stage B complete: {len(results)} papers debated")
        return results


# ── Stage C: Synthesis & Report ─────────────────────────────────────────────

class StageC:
    """Cluster papers → write theme chapters → edit intro → render report."""

    def __init__(self, agent: AIAgent, output_dir: Path):
        self.agent = agent
        self.output_dir = output_dir
        self.report_file = output_dir / "report.md"

    def _cluster(self, stage_a_results: list[StageAResult]) -> ClusterResult:
        papers_tags = "\n".join(
            f"- {r.arxiv_id} | {r.title} | tags: {', '.join(r.topics.tags)}"
            for r in stage_a_results
        )
        prompt = CLUSTERER_PROMPT.format(papers_tags=papers_tags)
        result = self.agent.structured(prompt, ClusterResult, mode="instruct_reasoning")
        if not isinstance(result, ClusterResult):
            result = ClusterResult(**result)
        return result

    def _write_theme(self, cluster, stage_a_results, stage_b_results) -> ThemeChapter:
        # Gather data for papers in this cluster
        a_lookup = {r.arxiv_id: r for r in stage_a_results}
        b_lookup = {r.arxiv_id: r for r in stage_b_results}

        papers_data_parts = []
        for pid in cluster.paper_ids:
            a = a_lookup.get(pid)
            b = b_lookup.get(pid)
            if a:
                part = (
                    f"### {a.title} ({pid})\n"
                    f"Problem: {a.extraction.problem}\n"
                    f"Method: {a.extraction.method}\n"
                    f"Novelty: {a.novelty.overall_novelty}/5\n"
                    f"Skeptic credibility: {a.skeptic.overall_credibility}/5\n"
                )
                if b:
                    part += (
                        f"Judge importance: {b.verdict.importance}/10\n"
                        f"Judge summary: {b.verdict.summary}\n"
                        f"Debate excerpt: {b.full_transcript[:500]}...\n"
                    )
                papers_data_parts.append(part)

        papers_data = "\n\n".join(papers_data_parts) if papers_data_parts else "No detailed data available."

        prompt = THEME_WRITER_PROMPT.format(
            theme_name=cluster.theme_name,
            theme_description=cluster.theme_description,
            papers_data=papers_data,
        )
        result = self.agent.structured(prompt, ThemeChapter, mode="instruct_reasoning")
        if not isinstance(result, ThemeChapter):
            result = ThemeChapter(**result)
        return result

    def _write_intro(self, theme_chapters, total_papers, deep_reviewed) -> ReportIntro:
        theme_summaries = "\n\n".join(
            f"## {ch.theme_name}\n{ch.overview}\nTop papers: {', '.join(ch.top_papers[:3])}"
            for ch in theme_chapters
        )
        prompt = EDITOR_PROMPT.format(
            date=datetime.now().strftime("%Y-%m-%d"),
            total_papers=total_papers,
            deep_reviewed=deep_reviewed,
            theme_summaries=theme_summaries,
        )
        result = self.agent.structured(prompt, ReportIntro, mode="instruct_reasoning")
        if not isinstance(result, ReportIntro):
            result = ReportIntro(**result)
        return result

    def _build_appendix(self, stage_a_results, stage_b_results) -> str:
        b_lookup = {r.arxiv_id: r for r in stage_b_results}
        lines = ["# Appendix: All Reviewed Papers\n"]
        for a in sorted(stage_a_results, key=lambda x: x.title):
            b = b_lookup.get(a.arxiv_id)
            importance = f" | Importance: {b.verdict.importance}/10" if b else ""
            verdict_line = b.verdict.summary[:150] if b else a.extraction.contribution[:150]
            lines.append(
                f"- **{a.title}** ([{a.arxiv_id}](https://arxiv.org/abs/{a.arxiv_id})){importance}\n"
                f"  {verdict_line}\n"
            )
        return "\n".join(lines)

    def _render_report(self, intro: ReportIntro, theme_chapters: list[ThemeChapter],
                       appendix: str, total_papers: int, deep_reviewed: int) -> str:
        parts = []
        # Cover
        parts.append(f"# ArXiv AI Daily Report — {datetime.now().strftime('%Y-%m-%d')}\n")
        parts.append(f"**{total_papers} papers analyzed | {deep_reviewed} deeply reviewed**\n")

        # TL;DR
        parts.append("## TL;DR — Today's Biggest Stories\n")
        for bullet in intro.tldr_bullets:
            parts.append(f"- {bullet}")
        parts.append("")

        # What to read first
        parts.append("## What to Read First\n")
        for pick in intro.what_to_read_first:
            parts.append(f"- {pick}")
        parts.append("")

        # Intro
        parts.append("## Introduction\n")
        parts.append(intro.intro_markdown)
        parts.append("")

        # Theme chapters
        for ch in theme_chapters:
            parts.append(f"## {ch.theme_name}\n")
            parts.append(ch.chapter_markdown)
            parts.append("")

        # Appendix
        parts.append(appendix)

        report = "\n\n".join(parts)
        return report

    def run(self, stage_a_results: list[StageAResult],
            stage_b_results: list[StageBResult]) -> str:
        """Full Stage C: cluster → themes → intro → render."""
        logger.info("Stage C: Clustering papers...")
        clusters = self._cluster(stage_a_results)
        logger.info(f"  Found {len(clusters.clusters)} clusters")

        # Write theme chapters in parallel
        logger.info("Stage C: Writing theme chapters...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(clusters.clusters), 10)) as pool:
            futures = [
                pool.submit(self._write_theme, c, stage_a_results, stage_b_results)
                for c in clusters.clusters
            ]
            theme_chapters = [f.result() for f in futures]
        logger.info(f"  Wrote {len(theme_chapters)} theme chapters")

        # Write intro
        logger.info("Stage C: Writing introduction...")
        intro = self._write_intro(
            theme_chapters,
            total_papers=len(stage_a_results),
            deep_reviewed=len(stage_b_results),
        )

        # Build appendix
        appendix = self._build_appendix(stage_a_results, stage_b_results)

        # Render
        logger.info("Stage C: Rendering final report...")
        report = self._render_report(
            intro, theme_chapters, appendix,
            total_papers=len(stage_a_results),
            deep_reviewed=len(stage_b_results),
        )

        with open(self.report_file, "w") as f:
            f.write(report)
        logger.info(f"Stage C complete: report saved to {self.report_file}")
        return report


# ── Pipeline Orchestrator ───────────────────────────────────────────────────

class ArxivPipeline:
    """Orchestrates Stages A, B, C with concurrent queue-driven execution."""

    def __init__(self, config: AIConfig = None, output_dir: str = None,
                 stage_a_workers: int = 50, stage_b_workers: int = 20,
                 download_rate_limit: float = 3.0):
        self.config = config or AIConfig()
        self.output_dir = Path(output_dir or f"arxiv_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.agent = AIAgent(config=self.config)
        self.stage_a = StageA(self.agent, self.output_dir, max_workers=stage_a_workers)
        self.stage_b = StageB(self.agent, self.output_dir, max_debate_workers=stage_b_workers,
                              download_rate_limit=download_rate_limit)
        self.stage_c = StageC(self.agent, self.output_dir)

    def run(self, papers: list[dict]) -> str:
        """Run the full pipeline. Returns path to the final report."""
        t_start = time.monotonic()
        read_queue = queue.Queue()

        logger.info(f"═══ ArXiv AI Pipeline ═══")
        logger.info(f"Papers: {len(papers)} | Output: {self.output_dir}")

        # Stage A + B run concurrently via queue
        stage_b_thread = threading.Thread(
            target=lambda: setattr(self, '_stage_b_results',
                                   self.stage_b.run(read_queue, [])),
            daemon=True,
        )

        # Start B consumer first (it blocks on queue)
        stage_b_thread.start()

        # Run A (producer) — this populates the queue
        stage_a_results = self.stage_a.run(papers, read_queue)

        # Update B with all stage A results for contextualizer
        self.stage_b._all_stage_a_results = stage_a_results

        # Wait for B to finish
        stage_b_thread.join()
        stage_b_results = getattr(self, '_stage_b_results', [])

        # Stage C: Synthesis
        report = self.stage_c.run(stage_a_results, stage_b_results)

        elapsed = round(time.monotonic() - t_start, 1)
        logger.info(f"═══ Pipeline complete in {elapsed}s ═══")
        logger.info(f"Report: {self.stage_c.report_file}")

        # Save metadata
        meta = {
            "total_papers": len(papers),
            "stage_a_processed": len(stage_a_results),
            "stage_b_debated": len(stage_b_results),
            "elapsed_seconds": elapsed,
            "output_dir": str(self.output_dir),
        }
        with open(self.output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        return str(self.stage_c.report_file)
