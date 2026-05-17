"""
arxiv-ai: fetch papers, run an agentic analysis on each PDF, email report.

Agent gets:
  - PDF as images (sees the full paper visually)
  - Tools: read_file, grep_file — operating on extracted text file
  - It grips exactly what it needs, works precisely

Usage:
    python main.py --limit 2 --categories AI
    python main.py --days 3 --limit 20
    python main.py --no-cache
"""

import argparse
import base64
import json
import os
import re
import subprocess
import tempfile
import time
import requests
import fitz
import pymupdf4llm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from openai import OpenAI
from gmaillite import gmail

from tool import fetch_latest_arxiv_cs_papers

BASE_URL = "http://192.168.170.49:8077/v1"
MODEL    = "/home/ng6355/models/qwen3-6-27b"
CLIENT   = OpenAI(base_url=BASE_URL, api_key="x")

SYSTEM_PROMPT = """\
You are an expert AI researcher analyzing a research paper.

You have:
1. First 15 pages of the paper as images (figures, tables, equations)
2. Three tools to work on the full extracted text:
   - get_toc(pdf_path) — get table of contents with page numbers. Call this first.
   - read_file(path, start_line, end_line) — read specific lines from text file
   - grep_file(path, pattern) — search for equations, keywords, section names

Workflow: get_toc → identify sections → read/grep the exact parts you need. Don't guess — grip precisely.

After your analysis, produce a final report:
1. **Problem** — what does it solve?
2. **Method** — exact technical approach, cite equations/figures/tables
3. **Results** — key numbers, are they credible?
4. **Novelty** — what's genuinely new?
5. **Critique** — real weaknesses
6. **Score** — X/10
7. **Verdict** — one sentence
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_toc",
            "description": "Get the table of contents of the paper — section titles and page numbers. Call this first to understand the paper structure before gripping specific sections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pdf_path": {"type": "string"},
                },
                "required": ["pdf_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read specific lines from the extracted paper text file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":       {"type": "string"},
                    "start_line": {"type": "integer", "description": "1-indexed start line"},
                    "end_line":   {"type": "integer", "description": "1-indexed end line"},
                },
                "required": ["path", "start_line", "end_line"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_file",
            "description": "Search for a pattern in the paper text file, returns matching lines with line numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string"},
                    "pattern": {"type": "string", "description": "text or regex to search for"},
                },
                "required": ["path", "pattern"],
            },
        },
    },
]


# ── Tool execution ────────────────────────────────────────────────────────────

def execute_tool(name, args):
    if name == "get_toc":
        try:
            doc = fitz.open(args["pdf_path"])
            toc = doc.get_toc()  # [[level, title, page], ...]
            if not toc:
                return "No TOC found in this PDF."
            lines = [f"{'  ' * (lvl-1)}[p{page}] {title}" for lvl, title, page in toc]
            return "\n".join(lines)
        except Exception as e:
            return f"error: {e}"

    if name == "read_file":
        path  = args.get("path") or args.get("file_path", "")
        start = max(1, args.get("start_line", 1)) - 1
        end   = args.get("end_line", start + 50)
        try:
            lines = open(path).readlines()
            chunk = lines[start:end]
            return "".join(f"{start+i+1}: {l}" for i, l in enumerate(chunk))
        except Exception as e:
            return f"error: {e}"

    elif name == "grep_file":
        path    = args.get("path") or args.get("file_path", "")
        pattern = args.get("pattern") or args.get("query") or args.get("search", "")
        try:
            result = subprocess.run(
                ["grep", "-n", "-i", pattern, path],
                capture_output=True, text=True, timeout=10
            )
            return result.stdout[:3000] or "no matches"
        except Exception as e:
            return f"error: {e}"

    return "unknown tool"


# ── PDF utilities ─────────────────────────────────────────────────────────────

def pdf_to_images(pdf_path, max_pages=15):
    doc    = fitz.open(pdf_path)
    blocks = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        b64 = base64.b64encode(pix.tobytes("jpeg")).decode()
        blocks.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return blocks


def pdf_to_textfile(pdf_path):
    """Extract PDF text to a temp file, return its path."""
    text = pymupdf4llm.to_markdown(pdf_path)
    tmp  = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    tmp.write(text)
    tmp.close()
    return tmp.name


def download_pdf(url, path, cache):
    if cache and os.path.exists(path):
        return True
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        time.sleep(3)
        return True
    except Exception as e:
        print(f"  download failed: {e}")
        return False


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(images, txt_path, pdf_path, max_steps=10):
    """Agentic loop: model sees PDF images, uses tools on txt_path, returns final answer."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    f"Analyze this paper thoroughly using your tools.\n"
                    f"- PDF path (for get_toc): {pdf_path}\n"
                    f"- Text file path (for read_file/grep_file): {txt_path}\n\n"
                    f"Start with get_toc to understand the structure, then read key sections, then write your full analysis ending with Score: X/10"
                )}
            ] + images
        }
    ]

    for _ in range(max_steps):
        resp = CLIENT.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=1500,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        msg         = resp.choices[0].message
        stop_reason = resp.choices[0].finish_reason

        messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in (msg.tool_calls or [])
        ]})

        if not msg.tool_calls:
            # no tool calls → model is done, return final text
            return msg.content or ""

        # Execute all tool calls
        for tc in msg.tool_calls:
            args   = json.loads(tc.function.arguments)
            print(f"    → {tc.function.name}({args})")
            result = execute_tool(tc.function.name, args)
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

    # Max steps hit — force final answer
    resp = CLIENT.chat.completions.create(
        model=MODEL,
        messages=messages + [{"role": "user", "content": "Now write your final analysis with Score: X/10"}],
        temperature=0.7,
        max_tokens=1500,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content or "no analysis"


# ── Per-paper entry point ─────────────────────────────────────────────────────

def analyze(paper, pdf_dir, cache):
    arxiv_id = paper["arxiv_id"].replace("/", "_")
    pdf_path = os.path.join(pdf_dir, f"{arxiv_id}.pdf")
    cached   = cache and os.path.exists(pdf_path)
    print(f"  {'[cache]' if cached else '[fetch]'} {paper['title'][:70]}")

    if not download_pdf(paper.get("pdf_url", ""), pdf_path, cache):
        return {"title": paper["title"], "arxiv_id": paper["arxiv_id"], "error": "download failed", "score": 0}

    try:
        images   = pdf_to_images(pdf_path)
        txt_path = pdf_to_textfile(pdf_path)
    except Exception as e:
        return {"title": paper["title"], "arxiv_id": paper["arxiv_id"], "error": str(e), "score": 0}

    try:
        # Prepend system prompt as first user turn (vLLM compatible)
        analysis = run_agent(images, txt_path, pdf_path)
    except Exception as e:
        analysis = f"agent error: {e}"
    finally:
        try:
            os.unlink(txt_path)
        except:
            pass

    m     = re.search(r'(\d+)\s*/\s*10', analysis)
    score = int(m.group(1)) if m else 0

    return {
        "title":    paper["title"],
        "arxiv_id": paper["arxiv_id"],
        "url":      paper.get("url", f"https://arxiv.org/abs/{paper['arxiv_id']}"),
        "analysis": analysis,
        "score":    score,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def build_report(results, date):
    ok     = sorted([r for r in results if "error" not in r], key=lambda x: x["score"], reverse=True)
    failed = [r for r in results if "error" in r]
    lines  = [f"# ArXiv AI Report — {date}", f"{len(ok)} papers | {len(failed)} failed\n"]
    for r in ok:
        lines += [f"## [{r['score']}/10] {r['title']}", f"{r['url']}\n", r["analysis"], "\n---\n"]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",       type=int, default=5)
    parser.add_argument("--categories", type=str, default=None)
    parser.add_argument("--limit",      type=int, default=20)
    parser.add_argument("--workers",    type=int, default=2)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--cache",      action="store_true", default=True)
    parser.add_argument("--no-cache",   dest="cache", action="store_false")
    args = parser.parse_args()

    date    = datetime.now().strftime("%Y-%m-%d")
    out_dir = args.output_dir or f"run_{date.replace('-','')}"
    pdf_dir = os.path.expanduser("~/.arxiv_pdf_cache")  # shared across all runs
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    cats = [c.strip() for c in args.categories.split(",")] if args.categories else None
    print("Fetching papers...")
    df = fetch_latest_arxiv_cs_papers(categories=cats, days=max(args.days, 5))
    if df is None or len(df) == 0:
        print("No papers found.")
        return

    papers = [p for p in df.to_dict("records") if p.get("pdf_url")][:args.limit]
    print(f"{len(papers)} papers\n")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(analyze, p, pdf_dir, args.cache): p for p in papers}
        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            results.append(r)
            print(f"  [{i}/{len(papers)}] score={r.get('score','err')} — {r['title'][:60]}")

    report = build_report(results, date)
    path   = os.path.join(out_dir, "report.md")
    open(path, "w").write(report)
    print(f"\nSaved: {path}")

    try:
        gmail(body=report, subject=f"ArXiv AI — {date} ({len(results)} papers)")
        print("Email sent.")
    except Exception as e:
        print(f"Email failed: {e}")


if __name__ == "__main__":
    main()
