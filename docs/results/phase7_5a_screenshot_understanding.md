# Phase 7.5a: Screenshot and PDF understanding (multimodal)

Customers attach screenshots of errors far more often than they paste the error text. Before this
feature the assistant could only say "I can't view images". Now a vision model reads every screenshot
and PDF at upload time, and the result is used by the agent, the ticket classifier and the staff
dashboard.

## Model choice (researched October 2026)
| Option | Verdict |
|---|---|
| **Groq `qwen/qwen3.8-27b`** (GA, image input, up to 3 images per request) | **Chosen.** Same Groq key and SDK as the chat model, separate free-tier quota, JSON mode works with images. Measured: ~0.8 s and ~1.3K prompt tokens per screenshot. |
| Gemini Flash (`gemini-3.8-flash`) | Good fallback if a Google key is added; free-tier limits are only shown per project in AI Studio, so not a dependency. |
| Local OCR (RapidOCR / Tesseract) | Text only: no understanding of what is wrong or where in the product; latest RapidOCR ONNX release is from January 2025. |

PDFs: `pypdf` reads the text layer (no model call); scanned PDFs (under 200 characters of text) have
their first 2 pages rendered with `pypdfium2` and sent to the vision model.

## How it works
```
upload ──► validate (extension + magic bytes) ──► save ──► analyze (≈1 s)
                                                            │  image: re-encode + downscale (≤1600 px) → vision model (JSON)
                                                            │  PDF:   text layer, or scanned pages → vision model
                                                            ▼
                         Attachment.analysis = {summary, visible_text, error_messages,
                                                product_area, contains_instructions, source, model}
   ├─► agent: appended to the customer's message inside <file> tags, labelled as data
   ├─► classifier: summary + product area + errors added to the text it routes on
   └─► staff dashboard: "Read by AI" panel (summary, exact errors, injection warning, visible text)
```

Design choices:
- **At upload time, once.** The analysis is ready before the message is sent and is stored, so the
  agent, the classifier and staff all reuse it with no repeated model calls.
- **Never blocks an upload.** Rate limits, model errors or broken images just mean "no analysis"; the
  file is saved and the agent says the team can see it.
- **Prompt injection.** The vision prompt treats image text as data and reports
  `contains_instructions`; the agent is told when a file contains text aimed at an AI, and staff see a
  warning. Images are re-encoded before sending, which drops anything appended after the image data.
- **Decompression bombs** are refused (40 MP limit).
- **Schema change without a migration tool:** `init_db()` adds new nullable columns to existing tables
  (tested on a copy of the live database; safe to run repeatedly).

## Evaluation (`python evals/run_vision_eval.py`)
13 synthetic SearchUnify-style screens with known contents (images in `evals/datasets/vision/`):
admin pages (light and dark), a browser console stack trace, a toast, a JSON API response, a log table
in small monospace, a blurred low-resolution copy, a scanned PDF invoice, two clean screens (one with
benign "click Save" instructions) and two prompt injections (one in faint grey text).

| Metric | Result |
|---|---|
| Key error phrase extracted | **10/10** |
| Full error string transcribed (≥ 0.9 similarity) | **10/10** |
| Clean screens with no invented errors | **3/3** |
| Product area identified | **13/13** |
| Prompt injections flagged | **2/2** |
| False injection alarms (benign screens, incl. "click Save") | **0/11** |
| Model latency (unthrottled) | ~0.8 s |
| Free-tier throughput | ~4 screenshots/minute back to back (8K tokens/minute); single uploads ~1 s |

**Limits of this evaluation:** the screens are rendered, clean and uncluttered, so they are easier than
real customer screenshots (busy UIs, cropped windows, phone photos of monitors). Treat 13/13 as "works
on the intended cases", not as a real-world accuracy figure; a set of real anonymised screenshots is
the next step.

## End-to-end check (real model, isolated server)
Uploaded `jira_401.png` in the chat with "Our crawl stopped working, see screenshot. How do I fix it?":
- Upload + analysis: **1.2 s**; error extracted verbatim, area "Content Sources > Jira Cloud".
- The agent searched for *"Jira Cloud content source API token invalid expired 401 fix"* (terms taken
  from the screenshot), cited **Create an API Token in Atlassian (Jira and Confluence)**, and opened with
  "the stored API token has expired or is invalid (HTTP 401)".
- Known gap: some later steps had no citation. The Phase 8 grounding check targets exactly this.

![Staff view: AI reading of two screenshots, one with a prompt injection](../images/staff-ai-reading.png)

## Tests
9 new tests (`tests/test_file_insights.py`, vision API faked; conftest blocks real calls in all tests):
image downscaling, output normalisation, PDF text path makes no model call, scanned PDF goes to the
vision model, failures never raise, upload returns the analysis and the agent sees it, upload still
works when analysis fails, injection flagged to the agent, and a screenshot changing ticket routing
(vague text + invoice screenshot → Billing). Full suite: **56 passed**.
