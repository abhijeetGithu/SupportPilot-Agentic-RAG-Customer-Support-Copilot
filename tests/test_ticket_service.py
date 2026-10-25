import pytest

from supportpilot.db.models import AuthorType, SessionMode, TicketSource
from supportpilot.tickets.schemas import TicketCreate, TicketUpdate
from supportpilot.tickets.service import TicketService, create_chat_session, mask_pii, parse_number
from tests.helpers import fake_classifier, word_overlap_similarity


def _svc(db):
    return TicketService(db, classifier=fake_classifier, similarity=word_overlap_similarity)


def _data(subject="Charged twice", description="I was charged twice for my plan this month.", email="Ana@Example.com"):
    return TicketCreate(customer_name="Ana", customer_email=email, subject=subject, description=description)


def test_create_classifies_and_logs_events(db):
    res = _svc(db).create(_data(), TicketSource.FORM)
    t = res.ticket
    assert t.number == "SP-00001" and t.category == "Billing and Payments" and t.priority == "high"
    assert t.customer_email == "ana@example.com" and not t.needs_review and not res.duplicate_of_existing
    ticket = _svc(db).get(t.id)
    assert [e.type for e in ticket.events] == ["created", "classified"]
    assert ticket.events[1].payload["model"] == "fake"


def test_low_confidence_needs_review(db):
    t = _svc(db).create(_data("Something odd", "The thing does not work the way I expected it to."), TicketSource.FORM)
    assert t.ticket.needs_review


def test_chat_ticket_stores_transcript(db):
    session = create_chat_session(db, "Ana", "ana@example.com")
    transcript = [(AuthorType.CUSTOMER, "I was charged twice"), (AuthorType.AI, "Sorry! Let me create a ticket.")]
    t = _svc(db).create(_data(), TicketSource.CHAT, session_id=session.id, transcript=transcript)
    ticket = _svc(db).get(t.ticket.id)
    assert [m.author_type for m in ticket.messages] == ["customer", "ai"]
    assert ticket.session_id == session.id


def test_duplicate_is_merged_not_created(db):
    svc = _svc(db)
    first = svc.create(_data(), TicketSource.CHAT)
    again = svc.create(_data(description="I was charged twice for my plan this month!"), TicketSource.FORM)
    assert again.duplicate_of_existing and again.ticket.id == first.ticket.id
    other_customer = svc.create(_data(email="bob@example.com"), TicketSource.FORM)
    assert not other_customer.duplicate_of_existing


def test_get_by_number_is_scoped_to_email(db):
    t = _svc(db).create(_data(), TicketSource.FORM).ticket
    assert _svc(db).get_by_number("SP-00001", "ana@example.com").id == t.id
    assert _svc(db).get_by_number("sp-1", "ANA@example.com").id == t.id
    assert _svc(db).get_by_number("SP-00001", "mallory@example.com") is None
    assert _svc(db).get_by_number("not a number") is None


def test_update_logs_reclassification_and_clears_review(db):
    svc = _svc(db)
    t = svc.create(_data("Something odd", "The thing does not work the way I expected it to."), TicketSource.FORM).ticket
    ticket = svc.update(t.id, TicketUpdate(category="IT Support", status="in_progress"), actor="staff")
    assert ticket.category == "IT Support" and ticket.status == "in_progress" and not ticket.needs_review
    events = {e.type: e.payload for e in ticket.events}
    assert events["reclassified"] == {"field": "category", "old": "Technical Support", "new": "IT Support"}
    assert events["status_changed"]["new"] == "in_progress"


def test_escalate_switches_session_to_human(db):
    session = create_chat_session(db, "Ana", "ana@example.com")
    t = _svc(db).create(_data(), TicketSource.CHAT, session_id=session.id).ticket
    ticket = _svc(db).escalate(t.id, "customer asked for a person")
    assert ticket.status == "escalated" and session.mode == SessionMode.HUMAN.value
    with pytest.raises(LookupError):
        _svc(db).escalate(999, "x")


def test_mask_pii_and_parse_number():
    assert mask_pii("card 4111 1111 1111 1111 please") == "card [card number] please"
    assert mask_pii("password: hunter2") == "password: [redacted]"
    assert parse_number("SP-01042") == 1042 and parse_number("sp1042") == 1042 and parse_number("1042x") is None
