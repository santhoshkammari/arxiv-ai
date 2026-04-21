● ArXiv AI — Thinking Out Loud                                                                                                                                                                  
                                                                                                                                                                                                
  OK. Dropping all the persistence/diff/temporal stuff. One-shot. Today's papers. Burn the GPU.                                                                                                 
                                                                                                                                                                                                
  ---                                                                                                                                                                                           
  The reframe                                                                                                                                                                                   
                                                                                                                                                                                                
  You have a local GPU that costs nothing per token. This changes everything. Every instinct I had in the last response about "be cheap at the abstract stage, be expensive only at finalists" —
   wrong. With Qwen3-30B on a 48GB card, tokens are free. The scarce resource is wall-clock time and arxiv's 3s rate limit, not tokens.                                                         
                                                                                                                                                                                                
  So the new law is: if the GPU isn't at 100% utilization, the pipeline is broken. Every idle millisecond is waste. We should have so many concurrent agents in flight that vLLM's batcher is   
  always full.                                                                                                                                                                                  
                                                                                                                                                                                                
  This flips the design. Instead of "filter cheap, debate expensive" we do deep treatment on every paper and let the synthesis stage sort it out.                                               
                                                                                                                                                                                                
  ---                                                                                                                                                                                           
  The core realization about concurrency                                                                                                                                                        
                                                                                                                                                                                                
  vLLM's superpower is continuous batching — 50 concurrent requests don't take 50× longer than 1, they take maybe 3–5× longer. So the right pattern is:                                         
                                                                                                                                                                                                
  - Don't process papers sequentially, ever                                                                                                                                                     
  - Don't process agents within a paper sequentially when they can be parallel                                                                                                                  
  - Fire everything you can into vLLM and let it batch                                                                                                                                          
                                                                                                                                                                                                
  For 1000 papers, the naive "one paper at a time with 5 agents" does 1000 sequential passes. The right version fires hundreds of concurrent requests and vLLM melts through them.              
                                                                                                                                                                                                
  Your AIAgent.batch() already does this with a ThreadPoolExecutor (ai.py:653). Good. We lean on that hard.                                                                                     
                                                 
  ---                                                                                                                                                                                           
  The pipeline, as I now see it                  
                                                                                                                                                                                                
  Three stages, fully overlapped via queues:
                                                                                                                                                                                                
  Stage A: Abstract Crunching        (pure vLLM, massively parallel, no I/O wait)
  Stage B: PDF Download + Deep Read  (I/O bound + vLLM, rate-limited I/O but parallel compute)                                                                                                  
  Stage C: Cross-paper Synthesis     (pure vLLM, reads outputs of A and B)                                                                                                                      
                                                                                                                                                                                                
  Stage A starts immediately after the CSV lands. Stage B starts the moment Stage A flags a paper worth PDF'ing (or for every paper — see below). Stage C starts as soon as Stage B outputs     
  start trickling in and keeps updating.                                                                                                                                                        
                                                                                                                                                                                                
  No stage waits for the previous to finish. All three are running simultaneously once the pipe is primed.                                                                                      
                                               
  ---                                                                                                                                                                                           
  Stage A: Abstract Crunching (the fan-out)      
                                               
  Every paper gets the same treatment pack in parallel. For each abstract, spawn these agents simultaneously (they don't need each other):
                                                                                                                                                                                                
  1. Extractor — pulls structured fields: problem, method, contribution, claimed results, datasets, benchmarks. Pydantic-constrained JSON output. This is the skeleton every later agent reads. 
  2. Novelty agent — "what's genuinely new here vs. reheated"? Generates 3 candidate "novel angles" and rates each 1–5.                                                                         
  3. Skeptic — attacks the abstract. "This claim is unsubstantiated", "this 'SOTA' ignores benchmark X", "this contribution is incremental". Produces an attack list.                           
  4. Topic tagger — fine-grained topic labels (not cs.AI broad categories — labels like rlhf-reward-modeling, long-context-attention, vision-language-grounding). These are what cluster on     
  later.                                                                                                                                                                                        
  5. Worth-reading-in-full? — binary gate with reasoning. Decides whether Stage B runs for this paper.                                                                                          
                                                                                                                                                                                                
  Concurrency: for 1000 papers × 5 agents = 5000 independent LLM calls. Pipe all of them through vLLM in waves of ~100 at a time. GPU saturated.                                                
                                                                                                                                                                                                
  Output of Stage A per paper: a JSON blob with extraction + novelty score + skeptic attacks + tags + read-gate decision. Save to stage_a.jsonl as it completes — streaming write, not batch at 
  end.                                           
                                                                                                                                                                                                
  ---                                            
  Stage B: PDF Download + Deep Read (the fight club)
                                                    
  This is where your vision lives. Two sub-stages, both running concurrently across papers.
                                                                                                                                                                                                
  B.1 — Downloader (I/O only, no GPU)                                                                                                                                                           
                                                                                                                                                                                                
  A single background worker consumes the "worth reading" queue from Stage A. Downloads PDFs respecting arxiv's 3s rate limit. As each PDF lands, extracts markdown + images via pymupdf4llm.   
  Pushes to a pdf_ready queue.                   
                                                                                                                                                                                                
  This is 1 thread doing ~20 papers/min. The GPU doesn't care — it's doing Stage A and Stage C work meanwhile.                                                                                  
                                               
  Images get extracted to disk. A fast multimodal captioning pass runs on each image (Qwen3.5-VL is multimodal — use it) and produces a 1–2 sentence caption. The markdown is rewritten so      
  ![](fig3.png) becomes ![fig3 — cross-attention layer with gated residual](fig3.png). Now text-only agents downstream get image semantics for free, and the final report can embed the actual
  images.                                                                                                                                                                                       
                                                 
  B.2 — Per-paper Debate (GPU goes brrr)                                                                                                                                                        
   
  For each PDF that lands, spawn a self-contained debate ensemble. This is per-paper; papers don't wait for each other; debates happen in parallel across papers.                               
                                                 
  The ensemble (all reading the same paper markdown + captioned images):                                                                                                                        
                                                 
  - Advocate — makes the strongest possible case. "Why is this important? What would change if this is true?"                                                                                   
  - Skeptic — attacks methodology, experimental design, baselines, claims. Writes specific objections.
  - Reproducer — "if I tried to implement this, what would I need? what's underspecified? what's the hidden cost?"                                                                              
  - Contextualizer — given the abstract-level extractions from other papers in Stage A, finds connections: "paper P42 claims the opposite", "paper P108 uses the same dataset"                  
  - Methodologist — focuses only on the technical method. Explains it precisely. If there's a diagram, references the captioned figure.                                                         
                                                                                                                                                                                                
  The debate mechanic — three rounds:                                                                                                                                                           
                                                                                                                                                                                                
  - Round 1: each agent produces their independent position (parallel — 5 concurrent calls)                                                                                                     
  - Round 2: each agent reads the other 4 positions and writes a rebuttal/update (parallel — 5 concurrent calls)
  - Round 3: a Judge agent reads the full transcript and produces the verdict: importance (1–10), confidence in claims (1–10), standout figure/result, open questions, a 200-word paper summary.
                                                                                                                                                                                                
  That's ~11 LLM calls per paper in Stage B. For 300 "worth reading" papers, 3300 calls. vLLM batches this like it's nothing.                                                                   
                                                                                                                                                                                                
  Why debate and not just "summarize": single-agent summaries regress to the mean. They paraphrase the abstract. Adversarial structure forces divergence — the Skeptic has to find real holes,  
  the Advocate has to defend, and that pressure surfaces the paper's actual structure in a way a summary agent never does. You've been dreaming of this for a reason.
                                                                                                                                                                                                
  Output per paper: the full transcript (keep it! it's gold) + the Judge's structured verdict.                                                                                                  
                                               
  ---                                                                                                                                                                                           
  Stage C: Cross-paper Synthesis (the report writer)
                                                                                                                                                                                                
  This runs continuously as Stage A and B outputs stream in. Not at the end.
                                                                                                                                                                                                
  C.1 — Clusterer. Takes all the topic tags from Stage A. Groups papers into themes. With ~1000 papers you'll get ~10–30 tight clusters. An LLM does the clustering (give it all tags, ask for  
  theme groupings) — embedding-based clustering is also fine but an LLM gives you named themes directly.                                                                                        
                                                                                                                                                                                                
  C.2 — Theme writer. One per cluster. Reads all Stage A extractions + all Stage B verdicts for papers in that cluster. Produces a 2-page theme chapter:                                        
  - What this theme is about                   
  - The 3–5 most important papers in it (using Judge importance scores)                                                                                                                         
  - Cross-paper tensions (where Skeptics attacked, where papers disagree)
  - Standout figures to embed                                                                                                                                                                   
  - Open questions the theme raises                                                                                                                                                             
                                                                                                                                                                                                
  C.3 — Editor. Reads all theme chapters. Writes the intro + the "across-themes" observations + the TOC + a "what to read first" page. Picks a cover figure.                                    
                                                                                                                                                                                                
  C.4 — Renderer. Stitches theme chapters into a single markdown doc with embedded images (from Stage B.1), then converts to PDF (pandoc or weasyprint). Target ~20 pages.                      
                                                                                                                                                                                                
  ---                                                                                                                                                                                           
  The concurrency architecture in one picture    
                                                                                                                                                                                                
       ┌───────────────┐
   CSV │ today's 1000  │                                                                                                                                                                        
       └──────┬────────┘                                                                                                                                                                        
              │                                
              ▼                                                                                                                                                                                 
     ┌────────────────────┐                      
     │  Stage A pool      │  ← 100 concurrent LLM calls                                                                                                                                         
     │  (5 agents/paper)  │     flood vLLM
     └────────┬───────────┘                                                                                                                                                                     
              │ streams JSONL                    
              ▼                                                                                                                                                                                 
     ┌────────────────────┐     ┌──────────────────┐
     │ read-gate filter   │────▶│  PDF downloader  │  ← rate-limited, 1 thread                                                                                                                  
     └────────────────────┘     │  (3s between)    │                                                                                                                                            
                                └────────┬─────────┘
                                         │ pdf_ready queue                                                                                                                                      
                                         ▼       
                                ┌────────────────────────┐                                                                                                                                      
                                │  Stage B pool          │  ← concurrent debates
                                │  (11 calls/paper)      │     across papers
                                └──────┬─────────────────┘                                                                                                                                      
                                       │ verdicts stream
                                       ▼                                                                                                                                                        
     ┌────────────────────┐   ┌────────────────────────┐
     │ Stage C clusterer  │──▶│ theme writers (parallel)│──▶ editor ──▶ PDF
     └────────────────────┘   └────────────────────────┘                                                                                                                                        
   
  All three stages alive simultaneously. Stage A finishes first (pure compute). Stage B keeps running until last PDF downloaded + debated. Stage C writes the report once B is substantially    
  done (wait for ~80% of debates, write preliminary, update at 100%).
                                                                                                                                                                                                
  ---                                                                                                                                                                                           
  Specific engineering choices                 
                                                                                                                                                                                                
  Why not read every PDF. Even with unlimited GPU, you're gated by arxiv's 3s download rate. 1000 papers × 3s = 50 minutes just for downloads. If Stage A's gate keeps ~30% of papers, that's 15
   minutes — much better, and the cut is smart not arbitrary. The Judge still gives us 300 debated papers to draw from; plenty for a 20-page report.                                            
   
  Structured output everywhere. Every agent returns pydantic-constrained JSON (your AIAgent.structured() already does this, ai.py:701). Prose is for the final report. Intermediate stages are  
  JSON.                                          
                                                                                                                                                                                                
  Save everything to disk as it completes. JSONL append-only files. Never hold 1000 papers' worth of agent outputs in memory. Lets you resume/debug, and lets Stage C start reading before Stage
   A is done.                                  
                                                                                                                                                                                                
  The debate transcript is a first-class artifact. Don't throw it away after extracting the verdict. The 20-page report quotes from it — "Skeptic agent noted: 'the authors' baseline is a 2020 
  model'". This is what makes the report feel considered rather than summarized.
                                                                                                                                                                                                
  Images via Qwen2.5-VL multimodal. The captioning pass is cheap and high-value. Do it. The final PDF embeds real figures with real captions — not just "Figure 3 shows...", but the actual     
  image the paper used.                        
                                                                                                                                                                                                
  Pool sizing. Stage A: 100 workers (threads). Stage B: 50 concurrent papers × but calls within a paper go in rounds so ~250 concurrent LLM calls at peak. vLLM batch size needs to be set high 
  (--max-num-seqs 256 or higher). If the GPU isn't at 100% during Stages A/B, increase pool size.
                                                                                                                                                                                                
  Stage C doesn't wait. It starts clustering on partial Stage A output (say, first 200 papers done) to produce a preliminary theme list. Gets refined as more data arrives. Final render waits  
  for everything.                              
                                                                                                                                                                                                
  ---                                            
  The "burn the GPU" checklist                 
                              
  - vLLM: max-num-seqs high, max-model-len big enough for PDF chunks
  - No time.sleep() anywhere in the agent code (the only sleep is the arxiv 3s rate limit on downloads, one thread)                                                                             
  - No synchronous "wait for PDF then run agent" — always queue-driven                                                                                                                          
  - No "batch at end" — everything streams to JSONL                                                                                                                                             
  - Pool sizes tuned so vLLM's queue depth stays ≥ 50 at all times                                                                                                                              
  - Stages A/B/C all alive at peak                                                                                                                                                              
  - Per-paper debates parallelize rounds (5 agents in a round → 5 concurrent calls, not 5 sequential)                                                                                           
                                                                                                                                                                                                
  ---                                                                                                                                                                                           
  What the final 20 pages look like                                                                                                                                                             
                                                                                                                                                                                                
  - Cover — date, paper count, headline themes, a cover figure from the most important paper
  - TL;DR page — 5 bullets: "today's biggest stories"                                                                                                                                           
  - What to read first — 5 picks with one-line pitches                                                                                                                                          
  - Theme chapters (×6–10) — each 1.5–2 pages: the theme, key papers, cross-paper tensions, a figure, open questions                                                                            
  - Adversarial highlights — a 1-page section with the spiciest Skeptic quotes and why they matter                                                                                              
  - Appendix index — all ~300 deeply-read papers with title, arxiv link, 2-line verdict                                                                                                         
                                                                                                                                                                                                
  That's the artifact. It's not a summary of 1000 papers. It's a curated, adversarially-vetted, thematically-structured read of today's research, with the agent debate visible enough that you 
  trust the judgment.                                                                                                                                                                           
                                                                                                                                                                                                

