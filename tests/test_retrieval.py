from llama_index.core.schema import TextNode

from supportpilot.ingestion.dedup import assign_dup_groups
from supportpilot.retrieval.retriever import collapse_duplicates
from supportpilot.retrieval.schemas import RetrievedChunk

BOILERPLATE = (
    "Return to the Content Sources screen and click Actions. Choose Crawl Frequency, then pick "
    "a date and an interval. The crawler will run on the schedule you selected every time."
)


def _node(cid: str, text: str) -> TextNode:
    return TextNode(id_=cid, text=text, metadata={})


def test_assign_dup_groups_links_copies_only():
    nodes = [
        _node("Jira.htm#3", BOILERPLATE),
        _node("Zendesk.htm#4", BOILERPLATE.replace("every time", "each time")),  # near copy
        _node("Jira.htm#0", "Jira needs an API token with Browse Users and Groups permission to connect."),
    ]
    assert assign_dup_groups(nodes, threshold=0.7) == 2
    assert nodes[0].metadata["dup_group"] == nodes[1].metadata["dup_group"] == "Jira.htm#3"
    assert nodes[0].metadata["dup_count"] == 2
    assert nodes[2].metadata["dup_group"] == "Jira.htm#0"
    assert nodes[2].metadata["dup_count"] == 1


def _hit(cid: str, group: str | None) -> RetrievedChunk:
    return RetrievedChunk(id=cid, doc_id=cid.split("#")[0], text="t", score=1.0, url="u", dup_group=group)


def test_collapse_duplicates_keeps_best_ranked_copy():
    hits = [_hit("A.htm#1", "g1"), _hit("B.htm#1", "g1"), _hit("C.htm#0", None), _hit("D.htm#2", "g1")]
    assert [h.id for h in collapse_duplicates(hits)] == ["A.htm#1", "C.htm#0"]


def test_rrf_fuse_weights_and_order():
    from supportpilot.retrieval.retriever import rrf_fuse

    dense = [_hit("A#0", None), _hit("B#0", None)]
    sparse = [_hit("B#0", None), _hit("C#0", None)]
    fused = rrf_fuse([(dense, 1.0), (sparse, 1.0)], k=2)
    assert [h.id for h in fused] == ["B#0", "A#0", "C#0"]  # B is in both lists
    assert abs(fused[0].score - (1 / 4 + 1 / 3)) < 1e-9
    dense_only = rrf_fuse([(dense, 1.0), (sparse, 0.0)], k=2)
    assert [h.id for h in dense_only][:2] == ["A#0", "B#0"]
