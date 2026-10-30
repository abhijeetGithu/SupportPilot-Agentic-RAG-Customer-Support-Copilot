"""Text helpers shared by ingestion and search, kept free of heavy imports."""


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
