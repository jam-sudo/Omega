"""Phase 43 — OAuth2/JWT authentication tests (14 unit tests)."""

from __future__ import annotations

import sys
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("passlib")
pytest.importorskip("jose")

# omega_pbpk.api.__init__ re-exports `app`, shadowing the submodule name,
# so we must retrieve the actual module via sys.modules.
import omega_pbpk.api.app  # noqa: F401 — ensure submodule is in sys.modules
import omega_pbpk.auth.jwt_auth as jwt_mod
from omega_pbpk.api.app import app  # type: ignore[attr-defined]

_app_module = sys.modules["omega_pbpk.api.app"]
client = TestClient(app)

_SIMULATE_BODY = {
    "drug": {"name": "Caffeine", "mw": 194.0, "logP": -0.07, "fup": 0.65, "rbp": 1.0},
    "dose_mg": 100.0,
    "route": "oral",
    "duration_h": 24.0,
}


@pytest.fixture(autouse=True)
def reset_users():
    jwt_mod.clear_users()
    yield
    jwt_mod.clear_users()


# ---------------------------------------------------------------------------
# 1. jwt_auth unit tests — passwords
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hash_and_verify_password():
    hashed = jwt_mod.hash_password("mysecret")
    assert jwt_mod.verify_password("mysecret", hashed) is True


@pytest.mark.unit
def test_verify_password_wrong():
    hashed = jwt_mod.hash_password("correct")
    assert jwt_mod.verify_password("wrong", hashed) is False


# ---------------------------------------------------------------------------
# 2. jwt_auth unit tests — tokens
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_create_access_token_valid():
    token = jwt_mod.create_access_token({"sub": "alice", "scopes": ["read"]})
    td = jwt_mod.verify_token(token)
    assert td is not None
    assert td.username == "alice"
    assert "read" in td.scopes


@pytest.mark.unit
def test_verify_token_expired():
    token = jwt_mod.create_access_token({"sub": "alice"}, expires_delta=timedelta(seconds=-1))
    assert jwt_mod.verify_token(token) is None


@pytest.mark.unit
def test_verify_token_invalid():
    assert jwt_mod.verify_token("not.a.valid.token") is None


# ---------------------------------------------------------------------------
# 3. jwt_auth unit tests — user store
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_create_user_and_get_user():
    user = jwt_mod.create_user("alice", "pass123")
    stored = jwt_mod.get_user("alice")
    assert stored is not None
    assert stored.username == "alice"
    assert stored.hashed_password == user.hashed_password


@pytest.mark.unit
def test_create_user_duplicate(monkeypatch):
    """Duplicate username: jwt_auth silently overwrites; HTTP endpoint raises 409."""
    monkeypatch.setattr(jwt_mod, "AUTH_ENABLED", True)
    monkeypatch.setattr(_app_module, "AUTH_ENABLED", True)

    client.post("/auth/register", json={"username": "carol", "password": "pw1"})
    resp = client.post("/auth/register", json={"username": "carol", "password": "pw2"})
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# 4. HTTP auth endpoint tests — auth enabled
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_register_endpoint(monkeypatch):
    monkeypatch.setattr(jwt_mod, "AUTH_ENABLED", True)
    monkeypatch.setattr(_app_module, "AUTH_ENABLED", True)

    resp = client.post("/auth/register", json={"username": "alice", "password": "secret"})
    assert resp.status_code == 200
    assert resp.json()["username"] == "alice"


@pytest.mark.unit
def test_login_endpoint(monkeypatch):
    monkeypatch.setattr(jwt_mod, "AUTH_ENABLED", True)
    monkeypatch.setattr(_app_module, "AUTH_ENABLED", True)

    client.post("/auth/register", json={"username": "alice", "password": "secret"})
    resp = client.post("/auth/token", data={"username": "alice", "password": "secret"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


@pytest.mark.unit
def test_login_invalid_credentials(monkeypatch):
    monkeypatch.setattr(jwt_mod, "AUTH_ENABLED", True)
    monkeypatch.setattr(_app_module, "AUTH_ENABLED", True)

    client.post("/auth/register", json={"username": "alice", "password": "correct"})
    resp = client.post("/auth/token", data={"username": "alice", "password": "wrong"})
    assert resp.status_code == 401


@pytest.mark.unit
def test_me_with_valid_token(monkeypatch):
    monkeypatch.setattr(jwt_mod, "AUTH_ENABLED", True)
    monkeypatch.setattr(_app_module, "AUTH_ENABLED", True)

    client.post("/auth/register", json={"username": "bob", "password": "hunter2"})
    login = client.post("/auth/token", data={"username": "bob", "password": "hunter2"})
    token = login.json()["access_token"]

    resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["username"] == "bob"


@pytest.mark.unit
def test_me_without_token(monkeypatch):
    monkeypatch.setattr(jwt_mod, "AUTH_ENABLED", True)
    monkeypatch.setattr(_app_module, "AUTH_ENABLED", True)

    resp = client.get("/auth/me")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 5. Protected endpoint tests — /simulate
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_protected_endpoint_auth_enabled_no_token(monkeypatch):
    monkeypatch.setattr(jwt_mod, "AUTH_ENABLED", True)
    monkeypatch.setattr(_app_module, "AUTH_ENABLED", True)

    resp = client.post("/simulate", json=_SIMULATE_BODY)
    assert resp.status_code == 401


@pytest.mark.unit
def test_protected_endpoint_auth_disabled():
    resp = client.post("/simulate", json=_SIMULATE_BODY)
    assert resp.status_code == 200
