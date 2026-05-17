"""
arxiv-ai: fetch today's papers, analyze each PDF with multimodal LLM, email report.

Two-pass per paper:
  SEEING  — PDF pages as images → model gets full visual context
  WORKING — extracted markdown text → model grips specific sections, works precisely

Usage:
    python main.py                        # last 5 days, all CS categories, limit 20
    python main.py --days 1 --limit 5     # test with 5 papers
    python main.py --categories AI,CL     # specific categories
    python main.py --cache                # reuse already-downloaded PDFs
"""

import argparse
import base64
import os
import re
import time
import requests
import fitz        # pymupdf
import pymupdf4llm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from openai import OpenAI
from gmaillite import gmail

from tool import fetch_latest_arxiv_cs_papers

BASE_URL = "http://192.168.170.49:8077/v1"
MODEL    = "/home/ng6355/models/qwen3-6-27b"
CLIENT   = OpenAI(base_url=BASE_URL, api_key="x")

# ── Prompts ──────────────────────────────────────────────────────────────────

SEEING_PROMPT = """\
You are an expert AI researcher. You are looking at the full PDF of a research paper.

Your job in this SEEING pass:
1. Identify the core problem, method, and key result — from what you visually see (figures, tables, equations).
2. Flag which sections/figures matter most for deep analysis.
3. Give an initial score 1-10. Format exactly as: Score: X/10
4. Decide: is this paper worth a deep working analysis? Answer YES or NO with one-line reason.

Be brief. This is triage, not the full review."""

WORKING_PROMPT = """\
You are an expert AI researcher doing a deep analysis of this paper.

Below is the full extracted text of the paper (markdown format).
You can read, reference, and reason over specific lines, equations, and tables precisely.

PAPER TEXT:
{text}

---

Now write a complete expert analysis:
1. **Problem**: What specific problem does this solve?
2. **Method**: Core technical approach — cite specific equations, algorithms, table numbers.
3. **Results**: Key claimed results with actual numbers. Are they credible?
4. **Novelty**: What is genuinely new? What does it borrow?
5. **Critique**: Real weaknesses — missing baselines, suspicious numbers, narrow scope, reproducibility.
6. **Score**: Rate 1-10. Format exactly as: Score: X/10
7. **Verdict**: One sentence. Worth reading or not, and exactly why.

Be direct, specific, critical. Reference actual content from the text above."""


# ── PDF utilities ─────────────────────────────────────────────────────────────

def pdf_to_images(pdf_path: str) -> list:
    """PDF pages → list of base64 image_url content blocks."""
    doc = fitz.open(pdf_path)
    content = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        b64 = base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    return content


def pdf_to_text(pdf_path: str, max_chars: int = 40000) -> str:
    """PDF → extracted markdown text, truncated to max_chars."""
    try:
        text = pymupdf4llm.to_markdown(pdf_path)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[... truncated ...]"
        return text
    except Exception as e:
        return f"[text extraction failed: {e}]"


def download_pdf(pdf_url: str, dest: str, cache: bool = True) -> bool:
    if cache and os.path.exists(dest):
        return True
    try:
        r = requests.get(pdf_url, stream=True, timeout=60)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


# ── LLM calls ────────────────────────────────────────────────────────────────

def llm_call(messages: list, max_tokens: int = 512) -> str:
    try:
        resp = CLIENT.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[LLM error: {e}]"


def extract_score(text: str) -> int:
    m = re.search(r'(?i)score\s*[:\-]?\s*(\d+)\s*/\s*10', text)
    return int(m.group(1)) if m else 0


# ── Core: analyze one paper ───────────────────────────────────────────────────

def analyze_paper(paper: dict, pdf_dir: str, cache: bool = True) -> dict:
    arxiv_id  = paper["arxiv_id"].replace("/", "_")
    pdf_path  = os.path.join(pdf_dir, f"{arxiv_id}.pdf")
    title     = paper["title"]

    # Download
    print(f"  {'[cached]' if (cache and os.path.exists(pdf_path)) else 'Downloading'}: {title[:60]}")
    if not download_pdf(paper.get("pdf_url", ""), pdf_path, cache=cache):
        return {"title": title, "arxiv_id": paper["arxiv_id"], "error": "download failed", "score": 0}

    if not cache:
        time.sleep(3)  # arxiv rate limit — only when actually downloading

    # ── SEEING pass (vision) ──────────────────────────────────────────────
    try:
        images = pdf_to_images(pdf_path)
    except Exception as e:
        return {"title": title, "arxiv_id": paper["arxiv_id"], "error": f"PDF open failed: {e}", "score": 0}

    seeing_out = llm_call([{
        "role": "user",
        "content": [{"type": "text", "text": SEEING_PROMPT}] + images
    }], max_tokens=400)

    initial_score  = extract_score(seeing_out)
    worth_working  = "YES" in seeing_out.upper()

    # ── WORKING pass (text grip) — only if seeing said worth it ──────────
    if worth_working:
        text = pdf_to_text(pdf_path)
        working_out = llm_call([{
            "role": "user",
            "content": WORKING_PROMPT.format(text=text)
        }], max_tokens=1200)
        final_score  = extract_score(working_out)
        analysis     = f"### SEEING\n{seeing_out}\n\n### WORKING (deep analysis)\n{working_out}"
        score        = final_score or initial_score
    else:
        analysis = f"### SEEING (skipped deep analysis)\n{seeing_out}"
        score    = initial_score

    return {
        "title":      title,
        "arxiv_id":   paper["arxiv_id"],
        "url":        paper.get("url", f"https://arxiv.org/abs/{paper['arxiv_id']}"),
        "analysis":   analysis,
        "score":      score,
        "deep":       worth_working,
    }


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(results: list, date: str) -> str:
    ok     = sorted([r for r in results if "error" not in r], key=lambda x: x.get("score", 0), reverse=True)
    failed = [r for r in results if "error" in r]

    lines = [
        f"# ArXiv AI Daily Report — {date}",
        f"**{len(ok)} papers analyzed** | {sum(1 for r in ok if r.get('deep'))} deep | {len(failed)} failed\n",
    ]

    top  = [r for r in ok if r.get("score", 0) >= 7]
    rest = [r for r in ok if r.get("score", 0) < 7]

    if top:
        lines.append("## TOP PICKS (Score ≥ 7)\n")
        for r in top:
            tag = "DEEP" if r.get("deep") else "SKIM"
            lines.append(f"### [{r['score']}/10] [{tag}] {r['title']}")
            lines.append(f"<{r['url']}>\n")
            lines.append(r["analysis"])
            lines.append("\n---\n")

    if rest:
        lines.append("## OTHER PAPERS\n")
        for r in rest:
            tag = "DEEP" if r.get("deep") else "SKIM"
            lines.append(f"### [{r.get('score','?')}/10] [{tag}] {r['title']}")
            lines.append(f"<{r['url']}>\n")
            lines.append(r["analysis"])
            lines.append("\n---\n")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",       type=int, default=5)
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--limit",      type=int, default=20)
    parser.add_argument("--workers",    type=int, default=3)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--cache",      action="store_true", default=True,
                        help="Reuse already-downloaded PDFs (default: on)")
    parser.add_argument("--no-cache",   dest="cache", action="store_false")
    args = parser.parse_args()

    date    = datetime.now().strftime("%Y-%m-%d")
    out_dir = args.output_dir or f"run_{date.replace('-', '')}"
    pdf_dir = os.path.join(out_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)

    cats = [c.strip() for c in args.categories.split(",")] if args.categories else None
    print(f"Fetching papers (days={args.days}, categories={cats or 'all'})...")
    df = fetch_latest_arxiv_cs_papers(categories=cats, days=max(args.days, 5))
    if df is None or len(df) == 0:
        print("No papers found.")
        return

    papers = [p for p in df.to_dict("records") if p.get("pdf_url")]
    if args.limit:
        papers = papers[:args.limit]
    print(f"Processing {len(papers)} papers (cache={'on' if args.cache else 'off'})...\n")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(analyze_paper, p, pdf_dir, args.cache): p for p in papers}
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            results.append(r)
            deep_tag = "[DEEP]" if r.get("deep") else "[SKIM]"
            print(f"  [{i}/{len(papers)}] {deep_tag} score={r.get('score','err')} — {r['title'][:55]}")

    report      = build_report(results, date)
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")

    try:
        gmail(body=report, subject=f"ArXiv AI Report — {date} ({len(results)} papers)")
        print("Email sent.")
    except Exception as e:
        print(f"Email failed: {e}")

    print("\nTop papers:")
    for r in sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:5]:
        print(f"  [{r.get('score','?')}/10] {r['title'][:70]}")


if __name__ == "__main__":
    main()
