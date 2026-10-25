from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    id: str  # chunk_id, e.g. "Content-Sources/Jira.htm#2"
    doc_id: str
    text: str
    score: float
    url: str
    title: str | None = None
    section: str | None = None
    doc_type: str | None = None
    heading: str | None = None
    dup_group: str | None = None  # chunks with the same dup_group are near-identical copies
