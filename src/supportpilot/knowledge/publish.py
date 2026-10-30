"""Index approved KB articles in Qdrant with the same chunking, contextual headers and dense + BM25
vectors as the docs, so hybrid search ranks them alongside the documentation with no other changes."""

from datetime import UTC, datetime

from qdrant_client import models

from supportpilot.config import get_settings
from supportpilot.db.models import KbArticle, KbStatus

SECTION = "Knowledge-Base"
DOC_TYPE = "kb"


def doc_id(article: KbArticle) -> str:
    return f"kb/{article.number}"


def article_url(article: KbArticle) -> str:
    return f"{get_settings().public_app_url.rstrip('/')}/support/kb/{article.number}"


def article_text(article: KbArticle) -> str:
    """What gets indexed: body plus keywords (synonyms customers use) for BM25 and embeddings."""
    keywords = ", ".join(article.keywords or [])
    area = f"Product area: {article.product_area}\n\n" if article.product_area else ""
    return f"# {article.title}\n\n{area}{article.body}" + (f"\n\nKeywords: {keywords}" if keywords else "")


def index_article(article: KbArticle, client=None, collection: str | None = None) -> int:
    """Chunk, embed and upsert; returns the number of chunks. Replaces any earlier version."""
    from llama_index.core import Document

    from supportpilot.ingestion.chunking import chunk_documents
    from supportpilot.retrieval.vector_store import get_client, upsert_chunks

    s = get_settings()
    client = client or get_client()
    collection = collection or s.qdrant_collection
    remove_article(article, client, collection)
    doc = Document(text=article_text(article), metadata={
        "doc_id": doc_id(article), "url": article_url(article), "title": article.title,
        "section": SECTION, "doc_type": DOC_TYPE,
    })
    chunks = chunk_documents([doc], s.chunk_size, s.chunk_overlap)
    upsert_chunks(client, collection, chunks)
    return len(chunks)


def remove_article(article: KbArticle, client=None, collection: str | None = None) -> None:
    from supportpilot.retrieval.vector_store import get_client

    client = client or get_client()
    client.delete(
        collection or get_settings().qdrant_collection,
        points_selector=models.FilterSelector(filter=models.Filter(must=[
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id(article)))
        ])),
    )


def publish(article: KbArticle, reviewer: str) -> KbArticle:
    index_article(article)
    article.status = KbStatus.PUBLISHED.value
    article.reviewed_by = reviewer
    article.published_at = datetime.now(UTC).replace(tzinfo=None)
    return article


def unpublish(article: KbArticle, reviewer: str) -> KbArticle:
    remove_article(article)
    article.status = KbStatus.DRAFT.value
    article.reviewed_by = reviewer
    article.published_at = None
    return article
