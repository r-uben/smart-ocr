# docr

Multi-engine OCR with cascading fallback, quality audit, and figure extraction.

Process academic papers and documents using free local models first, with automatic cloud fallback for failed pages. Extract and describe figures using vision models.

## Features

- **Multi-engine OCR** — Nougat, DeepSeek (Ollama), Mistral, Gemini
- **Smart routing** — Free local engines first, cloud only when needed
- **Quality audit** — Heuristics + LLM review catches garbage text
- **Figure extraction** — Renders figures from PDFs, describes with vision models
- **Batch processing** — Process entire directories of papers
- **CLI** — Live progress, colored panels, cost tracking

## Quick Start

```bash
# Install with Poetry
poetry install

# Install cloud engine SDKs (optional)
poetry install --extras cloud

# Process a paper
docr process paper.pdf

# Process with figure images saved
docr process paper.pdf --save-figures

# Batch process a folder
docr batch ~/Papers/ --limit 10
```

## Example Output

Processing a 22-page economics paper:

```
docr v0.1.0

kuttner_2001_monetary_policy.pdf
22 pages, 1.2 MB
type: academic

(1) primary ocr
    deepseek
    [+] page 1
    [+] page 2
    ...

(2) quality audit
    [!] page 10 (19.2% garbage)

(3) fallback ocr
    gemini
    [+] page 10

(4) figure processing
    [+] fig 1 (p.1): unknown
    [+] fig 2 (p.8): scatter_plot
    [+] fig 3 (p.12): scatter_plot

---

done 22/22 pages
     3 figures
     241.5s
     $0.0002
     deepseek (21) + gemini (1)

-> output/kuttner_2001_monetary_policy/kuttner_2001_monetary_policy.md
```

Output structure:
```
output/<doc_stem>/
├── <doc_stem>.md      # Full OCR text + figure descriptions
├── metadata.json      # Stats, engines used, cost
└── figures/           # With --save-figures
    ├── figure_1_page1.png
    └── ...
```

The `metadata.json` contains:
```json
{
  "document": "/path/to/paper.pdf",
  "stats": {
    "total_pages": 22,
    "pages_success": 22,
    "total_cost": 0.0002,
    "total_time": 241.5
  },
  "engines_used": { "deepseek": 21, "gemini": 1 },
  "figures": 3,
  "pages_needing_reprocessing": []
}
```

See `examples/kuttner_2001/` for a complete example with all 3 figures.

## Pipeline Architecture

```
                                    ┌─────────────┐
                                    │  PDF Input  │
                                    └──────┬──────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │  Document Classifier   │
                              │  (academic vs general) │
                              └───────────┬────────────┘
                                          │
          ┌───────────────────────────────┼───────────────────────────────┐
          │                               │                               │
          ▼                               ▼                               ▼
    ACADEMIC PATH                   GENERAL PATH                    ROUTER AGENT
 (Nougat → DeepSeek)             (DeepSeek → Nougat)            (picks available engine)
          │                               │                               │
          └───────────────────────────────┴───────────────────────────────┘
                                          │
                          ┌───────────────┴───────────────┐
                          │   STAGE 1: PRIMARY OCR        │
                          │                               │
                          │  ┌─────────────────────────┐  │
                          │  │ Local Engines (FREE):   │  │
                          │  │                         │  │
                          │  │  • Nougat               │  │
                          │  │    nougat_ocr lib       │  │
                          │  │    model: 0.1.0-small   │  │
                          │  │                         │  │
                          │  │  • DeepSeek             │  │
                          │  │    Ollama               │  │
                          │  │    deepseek-ocr:latest  │  │
                          │  └─────────────────────────┘  │
                          │             OR                │
                          │  ┌─────────────────────────┐  │
                          │  │ Cloud Engines (PAID):   │  │
                          │  │                         │  │
                          │  │  • Gemini               │  │
                          │  │    Google genai SDK     │  │
                          │  │    gemini-3-flash       │  │
                          │  │    ~$0.0002/page        │  │
                          │  │                         │  │
                          │  │  • Mistral              │  │
                          │  │    Mistral SDK          │  │
                          │  │    pixtral-large        │  │
                          │  │    ~$0.001/page         │  │
                          │  └─────────────────────────┘  │
                          └───────────────┬───────────────┘
                                          │
                                   [Page Results]
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │   STAGE 2: QUALITY AUDIT      │
                          │                               │
                          │  ┌─────────────────────────┐  │
                          │  │ Heuristics Checker      │  │
                          │  │ (rule-based, instant):  │  │
                          │  │  • Word count ≥ 50      │  │
                          │  │  • Garbage ratio < 15%  │  │
                          │  │  • Avg word length ok   │  │
                          │  │  • No unicode issues    │  │
                          │  │  • No repeated patterns │  │
                          │  └──────────┬──────────────┘  │
                          │             │                 │
                          │     [pages flagged?]          │
                          │             │                 │
                          │             ▼                 │
                          │  ┌─────────────────────────┐  │
                          │  │ Cross-Check (optional): │  │
                          │  │  Try 2nd local engine   │  │
                          │  └──────────┬──────────────┘  │
                          │             │                 │
                          │     [still flagged?]          │
                          │             │                 │
                          │             ▼                 │
                          │  ┌─────────────────────────┐  │
                          │  │ LLM Auditor (FREE):     │  │
                          │  │  Ollama                 │  │
                          │  │  llama3.2 or qwen2.5    │  │
                          │  │  • Can override heur.   │  │
                          │  │  • verdict: acceptable, │  │
                          │  │    needs_review, poor   │  │
                          │  └──────────┬──────────────┘  │
                          └─────────────┼─────────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                    [all pass]                   [some flagged]
                         │                             │
                         │                             ▼
                         │             ┌───────────────────────────────┐
                         │             │   STAGE 3: FALLBACK OCR       │
                         │             │                               │
                         │             │  Router selects different     │
                         │             │  engine (not primary):        │
                         │             │   Gemini → Mistral →          │
                         │             │   DeepSeek → Nougat           │
                         │             │                               │
                         │             │  Reprocess flagged pages      │
                         │             └───────────────┬───────────────┘
                         │                             │
                         └─────────────┬───────────────┘
                                       │
                                  [All Pages]
                                       │
                                       ▼
                          ┌────────────────────────────┐
                          │  STAGE 4: FIGURE AGENT     │
                          │  (if enabled)              │
                          │                            │
                          │  ┌──────────────────────┐  │
                          │  │ PyMuPDF Extractor:   │  │
                          │  │  • IMAGE blocks      │  │
                          │  │  • Embedded images   │  │
                          │  │  • Filter by size    │  │
                          │  │  • Render at 150 DPI │  │
                          │  └──────────┬───────────┘  │
                          │             │              │
                          │             ▼              │
                          │  ┌──────────────────────┐  │
                          │  │ Vision Engine:       │  │
                          │  │  Gemini 3 Flash      │  │
                          │  │  DeepSeek-OCR        │  │
                          │  │  Pixtral-Large       │  │
                          │  └──────────┬───────────┘  │
                          │             │              │
                          │      [Figure Results]      │
                          └─────────────┬──────────────┘
                                        │
                                        ▼
                          ┌─────────────────────────────┐
                          │      OUTPUT ASSEMBLY        │
                          │                             │
                          │  Format: markdown/json/txt  │
                          │                             │
                          │  output/<doc_stem>/         │
                          │    ├─ document.md           │
                          │    ├─ metadata.json         │
                          │    └─ figures/              │
                          │         └─ figure_N.png     │
                          └─────────────────────────────┘
```

**Cost optimization:**
- Local engines first (Nougat, DeepSeek via Ollama) - FREE
- Quality audit with local Ollama (llama3.2/qwen2.5) - FREE
- Cloud fallback only for failed pages (Gemini ~$0.0002/page)
- Typical 22-page paper: $0.0002 total (only 1 page needed cloud)

## Requirements

**Required:**
- Python 3.10+
- [Ollama](https://ollama.ai) with models:
  - `deepseek-ocr:latest` (primary OCR)
  - `llama3.2` or `qwen2.5` (quality audit)

**Optional (for cloud fallback):**
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- `MISTRAL_API_KEY`

```bash
# Install Ollama and pull models
ollama pull deepseek-ocr:latest
ollama pull llama3.2  # or qwen2.5

# Optionally pull Nougat for academic papers
# (requires nougat_ocr Python package)

# Check engine status
docr engines
```

## CLI Commands

```bash
# Process single PDF
docr process paper.pdf [OPTIONS]
  -o, --output PATH      Output file path
  -f, --format           markdown|json|txt
  --primary ENGINE       Force primary engine
  --fallback ENGINE      Force fallback engine
  --no-audit             Skip quality audit
  --no-figures           Skip figure processing
  --save-figures         Save figure images to disk

# Batch process directory
docr batch ~/Papers/ [OPTIONS]
  --limit N              Process first N files
  --save-figures         Save all figure images

# Check engines
docr engines

# Check audit system
docr audit-status
```

## Configuration

Create `docr.yaml` in your project or home directory:

```yaml
# Engine selection
primary_engine: deepseek
fallback_engine: gemini

# Quality audit
audit:
  enabled: true
  min_word_count: 50
  garbage_threshold: 0.15

# Figure processing
include_figures: true
save_figures: false
figures_max_total: 25
figures_max_per_page: 3

# Output
output_dir: output
output_format: markdown
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed module documentation.

```
src/docr/
├── cli.py              # Click CLI
├── core/
│   ├── config.py       # AgentConfig dataclass
│   ├── document.py     # PDF loading
│   └── result.py       # OCRResult, PageResult, FigureResult
├── engines/
│   ├── base.py         # BaseEngine ABC
│   ├── deepseek.py     # Ollama/DeepSeek
│   ├── gemini.py       # Google Gemini
│   ├── mistral.py      # Mistral AI
│   └── nougat.py       # Nougat (academic)
├── audit/
│   ├── heuristics.py   # Garbage detection
│   └── llm_audit.py    # Ollama-based review
├── pipeline/
│   ├── processor.py    # 4-stage pipeline
│   └── router.py       # Engine selection
└── ui/                 # Rich console output
```

## License

MIT
