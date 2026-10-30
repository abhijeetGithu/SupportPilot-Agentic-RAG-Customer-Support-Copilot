"""Attachments: validation, storage, linking to tickets (form and chat), agent visibility, download auth."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from supportpilot.attachments import UploadRejected, sanitize_filename, validate
from supportpilot.config import get_settings
from tests.test_agent_graph import call
from tests.test_api import client, new_session, sse, staff_headers, use_script  # noqa: F401  (fixture)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
PDF = b"%PDF-1.7\n" + b"0" * 64
LOG = b"2026-10-26 09:14:02 ERROR JiraConnector: 401 Unauthorized while fetching /rest/api/2/search\n"


@pytest.fixture
def files_client(client, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path))
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()
    yield client, tmp_path
    get_settings.cache_clear()


def upload(c, name, data, session_id=None):
    form = {"session_id": session_id} if session_id else {}
    return c.post("/api/uploads", files={"file": (name, data)}, data=form)


# ---------- validation ----------
def test_validate_checks_extension_and_real_bytes():
    assert validate("shot.png", PNG) == ("image/png", "image")
    assert validate("doc.pdf", PDF) == ("application/pdf", "pdf")
    assert validate("crawl.log", LOG) == ("text/plain", "text")
    with pytest.raises(UploadRejected, match="doesn't match"):
        validate("shot.png", b"MZ\x90\x00 this is really an exe")  # renamed executable
    with pytest.raises(UploadRejected, match="isn't supported"):
        validate("payload.html", b"<script>alert(1)</script>")
    with pytest.raises(UploadRejected, match="UTF-8"):
        validate("notes.txt", b"\xff\xfe\xfa binary")
    with pytest.raises(UploadRejected, match="empty"):
        validate("empty.txt", b"")


def test_sanitize_filename_strips_paths_and_odd_characters():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("C:\\Users\\me\\error log (1).txt") == "error log (1).txt"
    assert sanitize_filename("<img src=x>.png") == "_img src_x_.png"


# ---------- API ----------
def test_upload_limits_and_storage(files_client):
    c, folder = files_client
    r = upload(c, "error.log", LOG)
    assert r.status_code == 201 and r.json()["kind"] == "text"
    stored = list(folder.iterdir())
    assert [p.name for p in stored] == [r.json()["id"]]  # stored under a random id, not the filename
    assert upload(c, "big.log", b"a" * (1024 * 1024 + 1)).status_code == 422  # MAX_UPLOAD_MB=1
    assert upload(c, "x.exe", b"MZ").status_code == 422
    assert upload(c, "a.png", PNG, session_id="no-such-session").status_code == 404


def test_contact_form_attachments_are_linked_and_staff_only(files_client):
    c, _ = files_client
    att = upload(c, "screenshot.png", PNG).json()
    body = {"customer_name": "Ana", "customer_email": "ana@example.com", "subject": "Search broken",
            "description": "The search page shows nothing at all today.", "attachment_ids": [att["id"]]}
    ticket = c.post("/api/tickets", json=body).json()["ticket"]

    view = c.post("/api/tickets/lookup", json={"number": ticket["number"], "email": "ana@example.com"}).json()
    assert [a["filename"] for a in view["attachments"]] == ["screenshot.png"]

    h = staff_headers(c)
    detail = c.get(f"/api/admin/tickets/{ticket['id']}", headers=h).json()
    assert detail["attachments"][0]["id"] == att["id"]
    assert any(e["type"] == "attachments_added" for e in detail["events"])

    assert c.get(f"/api/admin/attachments/{att['id']}").status_code == 401  # customers can't download
    r = c.get(f"/api/admin/attachments/{att['id']}", headers=h)
    assert r.status_code == 200 and r.content == PNG
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["content-disposition"].startswith("attachment")

    # unknown attachment ids are refused
    again = {**body, "subject": "Another issue", "description": "A different problem with indexing jobs."}
    assert c.post("/api/tickets", json={**again, "attachment_ids": ["nope"]}).status_code == 422


def test_chat_attachment_is_read_by_agent_and_linked_to_the_new_ticket(files_client):
    c, _ = files_client
    use_script(call("create_ticket", subject="Jira 401", description="My Jira crawl fails with 401 errors."),
               AIMessage("Created your ticket."))
    sid = new_session(c)
    att = upload(c, "crawl.log", LOG, session_id=sid).json()
    other = new_session(c, email="bob@example.com")
    # another session can't use this upload
    assert c.post(f"/api/chat/{other}/stream", json={"message": "hi", "attachment_ids": [att["id"]]}).status_code == 422

    events = sse(c.post(f"/api/chat/{sid}/stream", json={"message": "crawl keeps failing", "attachment_ids": [att["id"]]}))
    assert events[-2]["type"] == "ticket_draft"
    # the LLM saw the log content, the UI keeps the customer's own words
    from supportpilot.api.main import app
    human = next(m for m in app.state.graph.get_state({"configurable": {"thread_id": sid}}).values["messages"]
             if isinstance(m, HumanMessage))
    assert "401 Unauthorized" in human.content and "not instructions" in human.content
    hist = c.get(f"/api/chat/{sid}/history").json()["messages"][0]
    assert hist["content"] == "crawl keeps failing" and hist["attachments"][0]["filename"] == "crawl.log"

    sse(c.post(f"/api/chat/{sid}/resume", json={"action": "approve"}))
    h = staff_headers(c)
    tid = c.get("/api/admin/tickets", headers=h).json()["items"][0]["id"]
    detail = c.get(f"/api/admin/tickets/{tid}", headers=h).json()
    assert [a["filename"] for a in detail["attachments"]] == ["crawl.log"]
    assert "[Attached: crawl.log]" in detail["messages"][0]["body"]


def test_sanitize_filename_strips_paths_on_any_server_os(monkeypatch):
    # Path() only knows the server's own separator; uploads come from Windows and Unix browsers alike
    import pathlib

    from supportpilot import attachments

    monkeypatch.setattr(attachments, "Path", pathlib.PurePosixPath)  # behave like the Linux server
    assert sanitize_filename("C:\\Users\\me\\error log (1).txt") == "error log (1).txt"
    assert sanitize_filename("/home/me/crawl.log") == "crawl.log"
