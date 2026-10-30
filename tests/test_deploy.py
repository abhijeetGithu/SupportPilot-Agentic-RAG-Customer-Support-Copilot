"""Public-deployment safeguards: rate limits, daily cap, read-only demo login, production config checks."""

from supportpilot.api import ratelimit
from supportpilot.config import Settings, check_production, get_settings
from tests.test_api import client, new_session, staff_headers  # noqa: F401  (fixture)


def enable_limits(monkeypatch, **env):
    monkeypatch.setenv("RATE_LIMITS", "true")
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    get_settings.cache_clear()
    ratelimit.reset()


def test_per_ip_rate_limit(client, monkeypatch):  # noqa: F811
    enable_limits(monkeypatch)
    body = {"name": "Ana", "email": "ana@example.com"}
    codes = [client.post("/api/sessions", json=body).status_code for _ in range(11)]
    assert codes[:10] == [201] * 10 and codes[10] == 429  # 10 sessions per hour per IP
    other_ip = client.post("/api/sessions", json=body, headers={"X-Forwarded-For": "203.0.113.9"})
    assert other_ip.status_code == 201  # limits are per client IP


def test_daily_cap_on_assistant_turns(client, monkeypatch):  # noqa: F811
    enable_limits(monkeypatch, DAILY_CHAT_LIMIT=2)
    sid = new_session(client)
    from langchain_core.messages import AIMessage

    from tests.test_api import use_script

    statuses = []
    for i in range(3):
        use_script(AIMessage("Hi!"))
        ip = {"X-Forwarded-For": f"198.51.100.{i}"}  # different IPs: only the global cap applies
        statuses.append(client.post(f"/api/chat/{sid}/stream", json={"message": "hello there"}, headers=ip).status_code)
    assert statuses == [200, 200, 429]


def test_demo_account_is_read_only(client, monkeypatch):  # noqa: F811
    monkeypatch.setenv("DEMO_PASSWORD", "look-around-only")
    get_settings.cache_clear()
    admin = staff_headers(client)
    t = client.post("/api/tickets", json={"customer_name": "Ana", "customer_email": "ana@example.com",
                                          "subject": "Search broken", "description": "The search page is blank."})
    tid = t.json()["ticket"]["id"]

    login = client.post("/api/auth/login", json={"username": "demo", "password": "look-around-only"}).json()
    assert login["role"] == "viewer"
    demo = {"Authorization": f"Bearer {login['access_token']}"}
    assert client.get(f"/api/admin/tickets/{tid}", headers=demo).status_code == 200
    blocked = client.patch(f"/api/admin/tickets/{tid}", json={"status": "resolved"}, headers=demo)
    assert blocked.status_code == 403 and "read-only" in blocked.json()["detail"]
    assert client.post(f"/api/admin/tickets/{tid}/replies", json={"body": "hi"}, headers=demo).status_code == 403
    assert client.patch(f"/api/admin/tickets/{tid}", json={"status": "resolved"}, headers=admin).status_code == 200


def test_demo_login_disabled_without_password(client):  # noqa: F811
    assert client.post("/api/auth/login", json={"username": "demo", "password": ""}).status_code == 401


def test_production_refuses_default_secrets():
    problems = check_production(Settings(environment="production", groq_api_key=""))
    assert len(problems) == 4
    ok = Settings(environment="production", jwt_secret="x" * 40, staff_password="a-long-password",
                  groq_api_key="gsk_test", database_url="postgresql+psycopg://u:p@db/sp")
    assert check_production(ok) == []


def test_forged_forwarded_for_cannot_dodge_limits(client, monkeypatch):  # noqa: F811
    enable_limits(monkeypatch)
    body = {"name": "Ana", "email": "ana@example.com"}
    # the proxy appends the real IP (198.51.100.7); the client-supplied first entry changes every time
    codes = [client.post("/api/sessions", json=body,
                         headers={"X-Forwarded-For": f"10.0.0.{i}, 198.51.100.7"}).status_code for i in range(11)]
    assert codes[-1] == 429
