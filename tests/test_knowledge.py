"""Knowledge loop: drafting from resolved tickets (anonymised), review and publishing, the public page,
question logging and the doc-gap report. LLM, Qdrant and embeddings are faked."""

import pytest
from langchain_core.messages import AIMessage

from supportpilot.db.models import KbArticle, QuestionLog
from supportpilot.knowledge import drafting, gaps, publish
from supportpilot.knowledge.drafting import Draft, scrub
from supportpilot.knowledge.gaps import gap_report, is_question
from tests.test_agent_graph import call
from tests.test_api import client, new_session, sse, staff_headers, use_script  # noqa: F401  (fixture)

DRAFT = Draft(
    title="Crawl stays in Queued after Ana Sharma at acme.com rotated the token",
    problem="The crawl for a content source stays in Queued and never starts. Reported in SP-00007.",
    symptoms=["Status: Queued"], cause="The crawler worker pool was paused after a token rotation.",
    resolution="1. Open **Content Sources**.\n2. Click **Stop**, then **Start**.",
    product_area="Content Sources", keywords=["queued", "crawl not starting", "ana@acme.com"],
    fixed_by="customer", note="Any customer can hit this after rotating a token.",
)


@pytest.fixture
def kb(client, monkeypatch):  # noqa: F811
    indexed, removed = [], []
    monkeypatch.setattr(drafting, "generate", lambda ticket, llm=None: (DRAFT.model_copy(), "fake-llm"))
    monkeypatch.setattr(gaps, "doc_coverage",
                        lambda text: {"title": "Add a Content Source", "url": "https://docs/x", "score": 0.61})
    monkeypatch.setattr(publish, "index_article", lambda a: indexed.append(a.number))
    monkeypatch.setattr(publish, "remove_article", lambda a: removed.append(a.number))
    return client, indexed, removed


def make_ticket(c, reply: str | None = "Please click Stop and then Start on the content source.",
                email: str = "ana@acme.com"):
    body = {"customer_name": "Ana Sharma", "customer_email": email, "subject": "Crawl stuck in queued",
            "description": "Our Zendesk crawl has been queued for 6 hours and never starts."}
    t = c.post("/api/tickets", json=body).json()["ticket"]
    h = staff_headers(c)
    if reply:
        c.post(f"/api/admin/tickets/{t['id']}/replies", json={"body": reply}, headers=h)
    return t, h


# ---------- drafting ----------
def test_scrub_removes_the_reporting_customer():
    class T:
        customer_name, customer_email = "Ana Sharma", "ana@acme.com"

    text = scrub("Ana Sharma (ana@acme.com) at Acme, see SP-00042; card 4111 1111 1111 1111", T())
    for leak in ("Ana", "Sharma", "ana@acme.com", "acme", "SP-00042", "4111"):
        assert leak.lower() not in text.lower()


def test_resolving_a_ticket_drafts_an_anonymised_article(kb):
    c, _, _ = kb
    t, h = make_ticket(c)
    c.patch(f"/api/admin/tickets/{t['id']}", json={"status": "resolved"}, headers=h)  # background task runs here
    [a] = c.get(f"/api/admin/kb?ticket_id={t['id']}", headers=h).json()
    assert a["status"] == "draft" and a["source_ticket_number"] == t["number"] and a["model"] == "fake-llm"
    everything = " ".join([a["title"], a["body"], " ".join(a["keywords"])]).lower()
    for leak in ("ana", "sharma", "acme", "sp-00007"):
        assert leak not in everything
    assert "## Resolution" in a["body"] and "**Stop**" in a["body"]
    assert a["doc_coverage"]["title"] == "Add a Content Source" and a["generalizable"] is True
    assert a["url"].endswith(f"/support/kb/{a['number']}")

    # resolving again doesn't create a second draft
    c.patch(f"/api/admin/tickets/{t['id']}", json={"status": "in_progress"}, headers=h)
    c.patch(f"/api/admin/tickets/{t['id']}", json={"status": "resolved"}, headers=h)
    assert len(c.get(f"/api/admin/kb?ticket_id={t['id']}", headers=h).json()) == 1
    assert c.post(f"/api/admin/tickets/{t['id']}/kb-draft", headers=h).status_code == 409


def test_no_draft_without_a_staff_reply(kb):
    c, _, _ = kb
    t, h = make_ticket(c, reply=None)
    c.patch(f"/api/admin/tickets/{t['id']}", json={"status": "resolved"}, headers=h)
    assert c.get(f"/api/admin/kb?ticket_id={t['id']}", headers=h).json() == []
    r = c.post(f"/api/admin/tickets/{t['id']}/kb-draft", headers=h)
    assert r.status_code == 409 and "no reply" in r.json()["detail"]


def test_drafting_failure_doesnt_break_resolving(client, monkeypatch):  # noqa: F811
    # conftest makes generate() raise, like a rate limit would
    t, h = make_ticket(client)
    r = client.patch(f"/api/admin/tickets/{t['id']}", json={"status": "resolved"}, headers=h)
    assert r.status_code == 200 and r.json()["ticket"]["status"] == "resolved"
    assert client.get("/api/admin/kb", headers=h).json() == []


def test_review_publish_and_public_page(kb):
    c, indexed, removed = kb
    t, h = make_ticket(c)
    a = c.post(f"/api/admin/tickets/{t['id']}/kb-draft", headers=h).json()
    assert c.get(f"/api/kb/{a['number']}").status_code == 404  # drafts aren't public

    edited = c.patch(f"/api/admin/kb/{a['id']}", json={"title": "Crawl stuck in Queued after a token rotation"},
                     headers=h).json()
    assert edited["title"] == "Crawl stuck in Queued after a token rotation" and edited["reviewed_by"] == "admin"
    assert indexed == []  # editing a draft doesn't touch the index

    pub = c.post(f"/api/admin/kb/{a['id']}/publish", headers=h).json()
    assert pub["status"] == "published" and pub["published_at"] and indexed == [a["number"]]
    page = c.get(f"/api/kb/{a['number'].lower()}").json()
    assert page["title"] == edited["title"] and "## Resolution" in page["body"]
    assert c.post(f"/api/admin/kb/{a['id']}/publish", headers=h).status_code == 409

    c.patch(f"/api/admin/kb/{a['id']}", json={"keywords": ["queued"]}, headers=h)
    assert indexed == [a["number"], a["number"]]  # edits to a published article are re-indexed

    c.post(f"/api/admin/kb/{a['id']}/reject", headers=h)
    assert removed == [a["number"]] and c.get(f"/api/kb/{a['number']}").status_code == 404
    assert c.get("/api/admin/kb?status=rejected", headers=h).json()[0]["id"] == a["id"]
    assert c.get("/api/admin/kb", headers={}).status_code == 401  # staff only


def test_index_article_uses_the_docs_pipeline(db, monkeypatch):
    from supportpilot.retrieval import vector_store

    upserted, deleted = [], []

    class FakeClient:
        def delete(self, collection, points_selector):
            deleted.append(points_selector.filter.must[0].match.value)

    monkeypatch.setattr(vector_store, "upsert_chunks", lambda qdrant, name, chunks: upserted.extend(chunks))
    a = KbArticle(id=3, title="Crawl stuck in Queued", body="## Problem\nIt stays queued.\n\n## Resolution\n1. Stop.",
                  product_area="Content Sources", keywords=["queued", "pending"])
    n = publish.index_article(a, client=FakeClient(), collection="test")
    meta = upserted[0].metadata
    assert n == len(upserted) >= 1 and deleted == ["kb/KB-00003"]  # old version removed first
    assert meta["doc_type"] == "kb" and meta["section"] == "Knowledge-Base" and meta["doc_id"] == "kb/KB-00003"
    assert meta["url"] == "http://localhost:3000/support/kb/KB-00003"
    assert "Keywords: queued, pending" in upserted[-1].text


def test_kb_passages_are_labelled_for_the_agent(monkeypatch):
    from supportpilot.agent import tools
    from supportpilot.retrieval.schemas import RetrievedChunk

    hit = RetrievedChunk(id="kb/KB-00003#0", doc_id="kb/KB-00003", text="Click Stop, then Start.", score=0.9,
                         url="http://localhost:3000/support/kb/KB-00003", title="Crawl stuck in Queued", doc_type="kb")
    monkeypatch.setattr(tools, "search", lambda q, expansions=None: [hit])
    out = tools.search_docs.func(query="crawl queued", state={"messages": []}, tool_call_id="1")
    assert out.update["messages"][0].content.startswith("[1] Support KB article: Crawl stuck in Queued")


# ---------- question logging and gaps ----------
def test_chat_questions_are_logged_with_outcomes(client, db, monkeypatch):  # noqa: F811
    monkeypatch.setattr(gaps, "doc_coverage", lambda text: {"title": "Jira", "url": "u", "score": 0.82})
    sid = new_session(client)
    use_script(AIMessage("Use an API token."))
    sse(client.post(f"/api/chat/{sid}/stream", json={"message": "How do I connect Jira to SearchUnify?"}))
    use_script(AIMessage("You're welcome!"))
    sse(client.post(f"/api/chat/{sid}/stream", json={"message": "thanks"}))  # not a question: not logged
    use_script(call("create_ticket", subject="Okta SSO", description="Okta SSO login fails for admins."))
    sse(client.post(f"/api/chat/{sid}/stream", json={"message": "Our Okta SSO login fails for every admin"}))

    logs = db.query(QuestionLog).order_by(QuestionLog.id).all()
    assert [(q.question, q.outcome, q.coverage) for q in logs] == [
        ("How do I connect Jira to SearchUnify?", "answered", 0.82),
        ("Our Okta SSO login fails for every admin", "ticket", 0.82),
    ]


def test_is_question():
    assert is_question("How do I add Jira?") and not is_question("thanks!") and not is_question("ok")
    assert not is_question("I've attached a file.")


def test_gap_report_groups_uncovered_questions(db):
    rows = [  # (question, coverage, outcome)
        ("How do I set up Okta SSO?", 0.55, "answered"), ("Okta SAML login fails for admins", 0.58, "ticket"),
        ("Okta single sign-on is broken", 0.60, "handoff"), ("Can I get a refund for unused seats?", 0.50, "answered"),
        ("How do I add Jira as a content source?", 0.85, "answered"),  # covered: not a gap
        ("Search results look odd after the update", 0.90, "ticket"),  # covered but needed a ticket: a gap
    ]
    for q, cov, outcome in rows:
        db.add(QuestionLog(session_id=None, question=q, coverage=cov, outcome=outcome, top_title="Doc", top_url="u"))
    db.flush()
    topic_of = {"okta": [1, 0, 0], "refund": [0, 1, 0], "update": [0, 0, 1]}

    def fake_embed(texts):
        return [next(v for k, v in topic_of.items() if k in t.lower()) for t in texts]

    r = gap_report(db, embed=fake_embed)
    assert r["questions"] == 6 and r["gap_questions"] == 5 and r["ended_in_ticket_or_handoff"] == 3
    top = r["topics"][0]
    assert top["count"] == 3 and top["label"] == "How do I set up Okta SSO?"
    assert top["outcomes"] == {"answered": 1, "ticket": 1, "handoff": 1} and top["avg_coverage"] == 0.577
    assert sorted(t["count"] for t in r["topics"]) == [1, 1, 3]


def test_fixes_done_by_support_are_flagged_as_not_reusable(kb, monkeypatch):
    c, _, _ = kb
    refund = DRAFT.model_copy(update={"fixed_by": "support_team", "note": "The support team issued the refund."})
    monkeypatch.setattr(drafting, "generate", lambda ticket, llm=None: (refund, "fake-llm"))
    t, h = make_ticket(c, reply="I've refunded the duplicate charge on your account.")
    a = c.post(f"/api/admin/tickets/{t['id']}/kb-draft", headers=h).json()
    assert a["generalizable"] is False and a["draft_note"] == "The support team issued the refund."


# ---------- linking articles in staff replies ----------
def test_agents_find_and_link_published_articles(kb, monkeypatch):
    from supportpilot.knowledge import related
    from supportpilot.retrieval import retriever
    from supportpilot.retrieval.schemas import RetrievedChunk

    c, _, _ = kb
    t, h = make_ticket(c)
    a = c.post(f"/api/admin/tickets/{t['id']}/kb-draft", headers=h).json()
    other, _ = make_ticket(c, email="bo@initech.com")
    draft_only = c.post(f"/api/admin/tickets/{other['id']}/kb-draft", headers=h)
    assert draft_only.status_code == 201  # a second, unpublished article
    c.post(f"/api/admin/kb/{a['id']}/publish", headers=h)

    searched = []

    def fake_dense(text, top_k, query_filter=None):
        searched.append((text, query_filter.must[0].match.value))
        chunk = lambda n, s: RetrievedChunk(id=f"kb/KB-{n:05d}#0", doc_id=f"kb/KB-{n:05d}", text="", score=s, url="u")
        return [chunk(a["id"], 0.71), chunk(a["id"], 0.65), chunk(draft_only.json()["id"], 0.9), chunk(99, 0.8)]

    monkeypatch.setattr(retriever, "dense_search", fake_dense)
    # suggestions for the ticket: only published articles, best chunk score, KB-only filter
    [s] = c.get(f"/api/admin/kb/search?ticket_id={t['id']}", headers=h).json()
    assert s["number"] == a["number"] and s["score"] == 0.71 and s["url"].endswith(f"/support/kb/{a['number']}")
    assert searched[0][1] == "kb" and searched[0][0].startswith("Crawl stuck in queued")
    # weak matches are hidden for automatic suggestions, shown for typed searches
    monkeypatch.setattr(related, "SUGGEST_MIN_SCORE", 0.8)
    assert c.get(f"/api/admin/kb/search?ticket_id={t['id']}", headers=h).json() == []
    assert len(c.get("/api/admin/kb/search?q=queued crawl", headers=h).json()) == 1
    assert c.get("/api/admin/kb/search", headers=h).status_code == 422

    # a staff reply with the link is tracked, and the Knowledge page shows the count
    c.post(f"/api/admin/tickets/{t['id']}/replies", headers=h,
           json={"body": f"This article explains it: [{s['title']}]({s['url']})"})
    detail = c.get(f"/api/admin/tickets/{t['id']}", headers=h).json()
    assert {"type": "kb_linked", "payload": {"articles": [a["number"]]}}.items() <= {
        k: v for e in detail["events"] if e["type"] == "kb_linked" for k, v in e.items()}.items()
    [listed] = c.get("/api/admin/kb?status=published", headers=h).json()
    assert listed["linked_in_replies"] == 1
