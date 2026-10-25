"""Load SearchUnify doc pages (MadCap Flare HTML) into LlamaIndex Documents.

Per page: keep only the article body (div#mc-main-content), turn screenshots into their
captions/alt text (the images themselves are not indexed), convert to markdown so headings
survive for chunking, and attach metadata used for citations and filtering.
"""

import hashlib
import re
from pathlib import Path

from bs4 import BeautifulSoup
from llama_index.core import Document
from markdownify import markdownify

BASE_URL = "https://docs.searchunify.com/Content/"

DOC_TYPES = {
    "Release-Notes": "release_notes",
    "FAQs": "faq",
    "Developer-Guides": "api",
    "API": "api",
}


def html_to_markdown(raw_html: str) -> tuple[str, str]:
    """Return (title, markdown body) for one Flare page."""
    soup = BeautifulSoup(raw_html, "lxml")
    main = soup.find(id="mc-main-content") or soup.body or soup
    for tag in main.find_all(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    for fig in main.find_all("figure"):
        caption = fig.find("figcaption")
        text = caption.get_text(" ", strip=True) if caption else ""
        if not text:
            img = fig.find("img")
            text = (img.get("alt") or "").strip() if img else ""
        fig.replace_with(soup.new_string(f"\n\n[Figure: {_strip_fig_prefix(text)}]\n\n") if text else "")
    for img in main.find_all("img"):  # inline images outside <figure>
        alt = (img.get("alt") or "").strip()
        img.replace_with(soup.new_string(f"[Image: {alt}]") if alt else "")

    h1 = main.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else (soup.title.get_text(strip=True) if soup.title else "")
    title = " ".join(title.split())  # some headings contain line breaks in the HTML

    md = markdownify(str(main), heading_style="ATX", strip=["a"], bullets="-")
    md = md.replace("\xa0", " ")
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return title, md


def _strip_fig_prefix(text: str) -> str:
    return re.sub(r"^\s*Fig(ure)?\.?\s*", "", text).strip()


def page_metadata(path: Path, source_dir: Path, title: str) -> dict:
    rel = path.relative_to(source_dir).as_posix()
    section = rel.split("/", 1)[0] if "/" in rel else "General"
    return {
        "doc_id": rel,
        "url": BASE_URL + rel,
        "title": title,
        "section": section,
        "page": path.stem,
        "doc_type": DOC_TYPES.get(section, "guide"),
    }


def load_documents(source_dir: Path, min_words: int = 80) -> tuple[list[Document], dict]:
    """Parse every .htm page; drop thin pages (navigation/index pages) and exact duplicates.

    Returns (documents, stats).
    """
    docs: list[Document] = []
    seen: set[str] = set()
    stats = {"pages": 0, "thin": 0, "duplicate": 0, "kept": 0}

    for path in sorted(source_dir.rglob("*.htm")):
        stats["pages"] += 1
        title, md = html_to_markdown(path.read_text(encoding="utf-8", errors="replace"))
        words = len(re.sub(r"\[(Figure|Image):[^\]]*\]", "", md).split())
        if words < min_words:
            stats["thin"] += 1
            continue
        digest = hashlib.md5(md.encode()).hexdigest()
        if digest in seen:
            stats["duplicate"] += 1
            continue
        seen.add(digest)

        meta = page_metadata(path, source_dir, title)
        docs.append(
            Document(
                text=md,
                doc_id=meta["doc_id"],
                metadata={**meta, "words": words},
                excluded_embed_metadata_keys=["url", "doc_id", "page", "words"],
                excluded_llm_metadata_keys=["doc_id", "page", "words"],
            )
        )
    stats["kept"] = len(docs)
    return docs, stats
