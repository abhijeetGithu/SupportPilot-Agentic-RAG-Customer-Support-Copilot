from pathlib import Path

from llama_index.core import Document

from supportpilot.ingestion.chunking import chunk_documents, count_tokens, embedding_text
from supportpilot.ingestion.loaders import html_to_markdown, load_documents

PAGE = """<html><head><title>t</title></head><body>
<nav>Site menu that must not be indexed</nav>
<div role="main" id="mc-main-content">
  <h1>Use Jira As a Content Source</h1>
  <p>Connect <a href="x.htm">Jira</a> to index issues.</p>
  <figure><p><img src="a.png" alt="alt text"></p>
    <figcaption><p>Fig. A snapshot of the Authentication tab.</p></figcaption></figure>
  <h2>Establish a Connection</h2>
  <p>Enter the client ID&#160;and secret.</p>
</div>
<div class="emailandfeedback">Did you find this page helpful?</div>
</body></html>"""


def test_html_to_markdown_keeps_article_only():
    title, md = html_to_markdown(PAGE)
    assert title == "Use Jira As a Content Source"
    assert "## Establish a Connection" in md
    assert "[Figure: A snapshot of the Authentication tab.]" in md
    assert "Site menu" not in md and "helpful" not in md
    assert "x.htm" not in md  # links stripped, text kept
    assert "Connect Jira to index" in md


def test_load_documents_metadata_and_thin_filter(tmp_path: Path):
    (tmp_path / "Content-Sources").mkdir()
    (tmp_path / "Content-Sources" / "Jira.htm").write_text(PAGE, encoding="utf-8")
    (tmp_path / "Content-Sources" / "Copy.htm").write_text(PAGE, encoding="utf-8")
    docs, stats = load_documents(tmp_path, min_words=5)
    assert stats == {"pages": 2, "thin": 0, "duplicate": 1, "kept": 1}
    meta = docs[0].metadata
    assert meta["section"] == "Content-Sources"
    assert meta["url"].startswith("https://docs.searchunify.com/Content/Content-Sources/")
    assert meta["doc_type"] == "guide"

    _, stats = load_documents(tmp_path, min_words=500)
    assert stats["thin"] == 2


def _doc(text: str) -> Document:
    return Document(
        text=text,
        metadata={"doc_id": "S/P.htm", "url": "u", "title": "Page", "section": "Search-Clients", "doc_type": "guide"},
    )


def test_chunks_respect_size_and_pack_small_sections():
    small = "\n\n".join(f"## Step {i}\n\nDo thing {i}." for i in range(6))
    big = "## Long\n\n" + " ".join(f"Sentence number {i} explains a setting." for i in range(400))
    chunks = chunk_documents([_doc("# Page\n\n" + small + "\n\n" + big)], chunk_size=128, chunk_overlap=16)
    assert all(count_tokens(c.text) <= 128 for c in chunks)
    assert "Step 0" in chunks[0].text and "Step 5" in chunks[0].text  # small sections packed together
    assert len(chunks) > 5  # big section split
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert chunks[0].id_ == "S/P.htm#0"


def test_embedding_text_has_context_header():
    meta = {"title": "Use Jira", "section": "Content-Sources", "heading": "Use Jira > Auth"}
    assert embedding_text(meta, "body") == "Use Jira > Content Sources > Auth\n\nbody"
    assert embedding_text({**meta, "heading": "Use Jira"}, "b") == "Use Jira > Content Sources\n\nb"


def test_heading_includes_own_section():
    doc = _doc("# Page\n\nIntro text here.\n\n## Setup\n\n" + "Setup details. " * 60)
    chunks = chunk_documents([doc], chunk_size=64, chunk_overlap=8)
    assert any(c.metadata["heading"] == "Page > Setup" for c in chunks)
