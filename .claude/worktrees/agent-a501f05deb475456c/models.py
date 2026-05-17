"""Pydantic models for all structured outputs in the ArXiv AI pipeline."""

from pydantic import BaseModel, Field
from typing import Optional


# ── Stage A Models ──────────────────────────────────────────────────────────

class ExtractionResult(BaseModel):
    """Structured extraction from a paper abstract."""
    problem: str = Field(description="The core problem the paper addresses")
    method: str = Field(description="The method or approach used")
    contribution: str = Field(description="Main contribution claimed")
    claimed_results: str = Field(description="Key results or improvements claimed")
    datasets: list[str] = Field(default_factory=list, description="Datasets or benchmarks mentioned")
    baselines: list[str] = Field(default_factory=list, description="Baseline methods compared against")


class NoveltyAngle(BaseModel):
    angle: str = Field(description="A novel angle or aspect of the paper")
    rating: int = Field(ge=1, le=5, description="Novelty rating 1-5")
    reasoning: str = Field(description="Why this rating")


class NoveltyResult(BaseModel):
    """Novelty assessment of a paper."""
    angles: list[NoveltyAngle] = Field(description="3 candidate novel angles with ratings")
    overall_novelty: int = Field(ge=1, le=5, description="Overall novelty score 1-5")
    summary: str = Field(description="Brief novelty summary")


class SkepticAttack(BaseModel):
    claim: str = Field(description="The specific claim being attacked")
    attack: str = Field(description="The skeptic's objection")
    severity: str = Field(description="low, medium, or high")


class SkepticResult(BaseModel):
    """Skeptic's attack list for a paper."""
    attacks: list[SkepticAttack] = Field(description="List of specific objections")
    overall_credibility: int = Field(ge=1, le=5, description="Overall credibility 1-5")


class TopicTags(BaseModel):
    """Fine-grained topic labels for a paper."""
    tags: list[str] = Field(description="Fine-grained topic labels like rlhf-reward-modeling, long-context-attention")


class ReadGate(BaseModel):
    """Binary gate: whether the paper is worth reading in full."""
    worth_reading: bool = Field(description="True if the paper is worth downloading and reading in full")
    reasoning: str = Field(description="Why or why not")


class StageAResult(BaseModel):
    """Combined output of all Stage A agents for one paper."""
    arxiv_id: str
    title: str
    extraction: ExtractionResult
    novelty: NoveltyResult
    skeptic: SkepticResult
    topics: TopicTags
    read_gate: ReadGate


# ── Stage B Models ──────────────────────────────────────────────────────────

class DebatePosition(BaseModel):
    """A single agent's position in the debate."""
    agent_name: str = Field(description="Name of the agent: Advocate, Skeptic, Reproducer, Contextualizer, Methodologist")
    position: str = Field(description="The agent's independent position or analysis")


class DebateRebuttal(BaseModel):
    """A single agent's rebuttal after reading other positions."""
    agent_name: str
    rebuttal: str = Field(description="Updated position after reading other agents")
    key_disagreements: list[str] = Field(default_factory=list, description="Key points of disagreement")


class JudgeVerdict(BaseModel):
    """Judge's final verdict on a paper after debate."""
    importance: int = Field(ge=1, le=10, description="Importance score 1-10")
    confidence_in_claims: int = Field(ge=1, le=10, description="How confident are we in the paper's claims 1-10")
    standout_result: str = Field(description="The standout figure or result")
    open_questions: list[str] = Field(description="Open questions raised by the debate")
    summary: str = Field(description="200-word paper summary incorporating debate insights")


class StageBResult(BaseModel):
    """Combined output of Stage B debate for one paper."""
    arxiv_id: str
    title: str
    round1_positions: list[DebatePosition]
    round2_rebuttals: list[DebateRebuttal]
    verdict: JudgeVerdict
    full_transcript: str = Field(description="Full debate transcript")


# ── Stage C Models ──────────────────────────────────────────────────────────

class PaperCluster(BaseModel):
    """A thematic cluster of papers."""
    theme_name: str = Field(description="Human-readable theme name")
    theme_description: str = Field(description="What this theme is about")
    paper_ids: list[str] = Field(description="ArXiv IDs of papers in this cluster")


class ClusterResult(BaseModel):
    """Output of the clustering step."""
    clusters: list[PaperCluster] = Field(description="Thematic clusters of papers")


class ThemeChapter(BaseModel):
    """A theme chapter for the final report."""
    theme_name: str
    overview: str = Field(description="What this theme is about")
    top_papers: list[str] = Field(description="3-5 most important paper titles with brief descriptions")
    cross_paper_tensions: str = Field(description="Where papers disagree or skeptics attacked")
    open_questions: list[str] = Field(description="Open questions the theme raises")
    chapter_markdown: str = Field(description="Full markdown text of the chapter")


class ReportIntro(BaseModel):
    """Editor's introduction and cross-theme observations."""
    headline_themes: list[str] = Field(description="Top headline themes for the day")
    tldr_bullets: list[str] = Field(description="5 TL;DR bullets: today's biggest stories")
    what_to_read_first: list[str] = Field(description="5 picks with one-line pitches")
    cross_theme_observations: str = Field(description="Observations that span multiple themes")
    intro_markdown: str = Field(description="Full markdown intro text")
