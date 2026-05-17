"""
arxiv-ai: fetch today's papers, analyze each PDF with multimodal LLM, email report.

Usage:
    python main.py                     # today, all CS categories, limit 20
    python main.py --days 1 --limit 5  # test with 5 papers
    python main.py --categories AI,CL  # specific categories
"""

import argparse
import base64
import os
import time
import requests
import fitz  # pymupdf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from openai import OpenAI
from gmaillite import gmail

from tool import fetch_latest_arxiv_cs_papers

BASE_URL = "http://192.168.170.49:8077/v1"
MODEL = "/home/ng6355/models/qwen3-6-27b"
CLIENT = OpenAI(base_url=BASE_URL, api_key="x")

ANALYSIS_PROMPT = """You are an expert AI researcher reviewing this paper.

Analyze it and answer:
1. **Problem**: What specific problem does this solve?
2. **Method**: What is the core technical approach? (reference actual figures/equations/tables you see)
3. **Results**: What are the key claimed results? Are the numbers credible?
4. **Novelty**: What is genuinely new here vs. existing work?
5. **Critique**: What are the weaknesses — missing baselines, suspicious claims, limited scope?
6. **Score**: Rate 1-10 (10 = must read, 1 = skip)
7. **Verdict**: One sentence: worth reading or not, and why.

Be direct, specific, and critical. Reference what you actually see in the PDF."""


def pdf_to_image_content(pdf_path: str) -> list:
    """Convert PDF pages to base64 image_url content blocks."""
    doc = fitz.open(pdf_path)
    content = []
    for page in doc:
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        b64 = base64.b64encode(pix.tobytes("jpeg")).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    return content


def download_pdf(pdf_url: str, dest: str) -> bool:
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


def analyze_paper(paper: dict, pdf_dir: str) -> dict:
    arxiv_id = paper["arxiv_id"].replace("/", "_")
    pdf_path = os.path.join(pdf_dir, f"{arxiv_id}.pdf")

    print(f"  Downloading: {paper['title'][:60]}...")
    if not download_pdf(paper["pdf_url"], pdf_path):
        return {"title": paper["title"], "arxiv_id": paper["arxiv_id"],
                "error": "Download failed", "score": 0}

    time.sleep(3)  # arxiv rate limit

    try:
        image_content = pdf_to_image_content(pdf_path)
    except Exception as e:
        return {"title": paper["title"], "arxiv_id": paper["arxiv_id"],
                "error": f"PDF parse failed: {e}", "score": 0}

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ANALYSIS_PROMPT}
            ] + image_content
        }
    ]

    try:
        resp = CLIENT.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        analysis = resp.choices[0].message.content
    except Exception as e:
        analysis = f"LLM call failed: {e}"

    # Extract score from analysis (look for "Score: X" pattern)
    score = 0
    for line in analysis.split("\n"):
        if "score" in line.lower() and any(c.isdigit() for c in line):
            digits = [c for c in line if c.isdigit()]
            if digits:
                score = int(digits[0])
                break

    return {
        "title": paper["title"],
        "arxiv_id": paper["arxiv_id"],
        "url": paper.get("url", f"https://arxiv.org/abs/{paper['arxiv_id']}"),
        "analysis": analysis,
        "score": score,
    }


def build_report(results: list, date: str) -> str:
    results_sorted = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
    ok = [r for r in results_sorted if "error" not in r]
    failed = [r for r in results_sorted if "error" in r]

    lines = [
        f"# ArXiv AI Daily Report — {date}",
        f"**{len(ok)} papers analyzed** | {len(failed)} failed\n",
    ]

    # Top picks
    top = [r for r in ok if r.get("score", 0) >= 7]
    if top:
        lines.append("## TOP PICKS (Score ≥ 7)\n")
        for r in top:
            lines.append(f"### [{r['score']}/10] {r['title']}")
            lines.append(f"<{r['url']}>\n")
            lines.append(r["analysis"])
            lines.append("\n---\n")

    # Rest
    rest = [r for r in ok if r.get("score", 0) < 7]
    if rest:
        lines.append("## ALL OTHER PAPERS\n")
        for r in rest:
            lines.append(f"### [{r.get('score', '?')}/10] {r['title']}")
            lines.append(f"<{r['url']}>\n")
            lines.append(r["analysis"])
            lines.append("\n---\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    date = datetime.now().strftime("%Y-%m-%d")
    out_dir = args.output_dir or f"run_{date.replace('-', '')}"
    pdf_dir = os.path.join(out_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)

    # Fetch papers
    cats = [c.strip() for c in args.categories.split(",")] if args.categories else None
    print(f"Fetching papers (days={args.days}, categories={cats or 'all'})...")
    df = fetch_latest_arxiv_cs_papers(categories=cats, days=args.days)
    papers = df.to_dict("records")

    # Filter papers with pdf_url
    papers = [p for p in papers if p.get("pdf_url")]
    if args.limit:
        papers = papers[:args.limit]

    print(f"Processing {len(papers)} papers...\n")

    # Analyze papers (parallel downloads + sequential LLM due to GPU memory)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(analyze_paper, p, pdf_dir): p for p in papers}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(f"  [{i}/{len(papers)}] {result['title'][:50]}... score={result.get('score', 'err')}")

    # Build report
    report = build_report(results, date)
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nReport saved: {report_path}")

    # Email
    try:
        gmail(
            body=report,
            subject=f"ArXiv AI Report — {date} ({len(results)} papers)",
        )
        print("Email sent.")
    except Exception as e:
        print(f"Email failed: {e}")

    print(f"\nDone. Top papers:")
    for r in sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:5]:
        print(f"  [{r.get('score', '?')}/10] {r['title'][:70]}")


if __name__ == "__main__":
    main()
