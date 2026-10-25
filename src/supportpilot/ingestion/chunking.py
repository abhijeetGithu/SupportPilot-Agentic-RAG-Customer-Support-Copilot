"""Split documents into retrieval chunks.

1. MarkdownNodeParser: one node per heading section (keeps the heading path).
2. Pack: merge consecutive small sections of the same page up to chunk_size tokens,
   so short "Step 3" sections don't become useless one-line chunks.
3. Split: sections larger than chunk_size go through SentenceSplitter with overlap.

Each chunk also gets a contextual header ("Page title > Section > Heading") that is
prepended only for embedding, so look-alike pages (e.g. 140 connector guides) stay
distinguishable.
"""

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.utils import get_tokenizer

_tokenizer = get_tokenizer()


def count_tokens(text: str) -> int:
    return len(_tokenizer(text))


def _heading(node) -> str:
    """Full heading path incl. the section's own heading (header_path only holds the parents)."""
    parents = [p for p in (node.metadata.get("header_path") or "").strip("/").split("/") if p]
    first_line = node.get_content().lstrip().split("\n", 1)[0]
    if first_line.startswith("#"):
        parents.append(first_line.lstrip("#").strip())
    return " > ".join(parents)


def _pack_sections(sections, chunk_size: int, splitter: SentenceSplitter) -> list[tuple[str, str]]:
    """Return (heading, text) pieces: small sections merged, large ones split."""
    pieces: list[tuple[str, str]] = []
    buf_text, buf_heading = "", ""
    for sec in sections:
        text = sec.get_content().strip()
        if not text:
            continue
        heading = _heading(sec)
        if count_tokens(text) > chunk_size:
            if buf_text:
                pieces.append((buf_heading, buf_text))
                buf_text = ""
            pieces.extend((heading, part) for part in splitter.split_text(text))
            continue
        candidate = f"{buf_text}\n\n{text}" if buf_text else text
        if buf_text and count_tokens(candidate) > chunk_size:
            pieces.append((buf_heading, buf_text))
            buf_text, candidate = "", text
        if not buf_text:
            buf_heading = heading
        buf_text = candidate
    if buf_text:
        pieces.append((buf_heading, buf_text))
    return pieces


def chunk_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[TextNode]:
    md_parser = MarkdownNodeParser()
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[TextNode] = []

    for doc in docs:
        pieces = _pack_sections(md_parser.get_nodes_from_documents([doc]), chunk_size, splitter)
        meta = doc.metadata
        for i, (heading, text) in enumerate(pieces):
            chunk_id = f"{meta['doc_id']}#{i}"
            chunks.append(
                TextNode(
                    id_=chunk_id,
                    text=text,
                    metadata={
                        "chunk_id": chunk_id,
                        "doc_id": meta["doc_id"],
                        "url": meta["url"],
                        "title": meta["title"],
                        "section": meta["section"],
                        "doc_type": meta["doc_type"],
                        "heading": heading,
                        "chunk_index": i,
                        "tokens": count_tokens(text),
                    },
                )
            )
    return chunks


def embedding_text(meta: dict, text: str) -> str:
    """Contextual chunk header + body. Used for both dense and BM25 vectors."""
    title = meta["title"]
    heading = meta.get("heading") or ""
    if heading == title:
        heading = ""
    elif heading.startswith(title + " > "):
        heading = heading[len(title) + 3 :]
    parts = [title, meta["section"].replace("-", " "), heading]
    return " > ".join(p for p in parts if p) + "\n\n" + text
