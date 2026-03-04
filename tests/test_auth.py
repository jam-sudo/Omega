"""Phase 43 — OAuth2/JWT authentication tests (14 unit tests)."""

from __future__ import annotations

import sys

import pytest
from fastapi.testclient import TestClient

# omega_pbpk.api.__init__ re-exports `app`, shadowing the submodule name,
# so we must retrieve the actual module via sys.modules.
import omega_pbpk.api.app  # noqa: F401 — ensure submodule is in sys.modules
import omega_pbpk.auth.jwt_auth as jwt_mod
from omega_pbpk.api.app import app  # type: ignore[attr-defined]

_app_module = sys.modules["omega_pbpk.api.app"]
client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_users():
    jwt_mod.clear_users()
    yield
    jwt_mod.clear_users()


# ---------------------------------------------------------------------------
# jwt_auth unit tests (no HTTP)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hash_and_verify_password():
    hashed = jwt_mod.hash_password("mysecret")
    assert jwt_mod.verify_password("mysecret", hashed) is True


@pytest.mark.unit
def test_verify_password_wrong():
    hashed = jwt_mod.hash_password("correct")
    assert jwt_mod.verify_password("wrong", hashed) is False


@pytest.mark.unit
def test_create_user_stored():
    user = jwt_mod.create_user("alice", "pass123")
    stored = jwt_mod.get_user("alice")
    assert stored is not None
    assert stored.username == "alice"
    assert stored.hashed_password == user.hashed_password


@pytest.mark.unit
def test_get_user_missing():
    assert jwt_mod.get_user("nobody") is None


@pytest.mark.unit
def test_clear_users():
    jwt_mod.create_user("bob", "pw")
    jwt_mod.clear_users()
    assert jwt_mod.get_user("bob") is None


@pytest.mark.unit
def test_create_access_token_returns_string():
    token = jwt_mod.create_access_token({"sub": "alice"})
    assert isinstance(token, str)
    assert len(token) > 10


@pytest.mark.unit
def test_verify_token_valid():
    token = jwt_mod.create_access_token({"sub": "alice", "scopes": ["read"]})
    td = jwt_mod.verify_token(token)
    assert td is not None
    assert td.username == "alice"
    assert "read" in td.scopes


@pytest.mark.unit
def test_verify_token_invalid():
    td = jwt_mod.verify_token("not.a.valid.token")
    assert td is None


# ---------------------------------------------------------------------------
# API endpoint tests — auth disabled (default)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_register_disabled_by_default():
    resp = client.post("/auth/register", json={"username": "alice", "password": "pw"})
    assert resp.status_code == 404
    assert "disabled" in resp.json()["detail"].lower()


@pytest.mark.unit
def test_token_disabled_by_default():
    resp = client.post("/auth/token", data={"username": "alice", "password": "pw"})
    assert resp.status_code == 404


@pytest.mark.unit
def test_me_disabled_by_default():
    resp = client.get("/auth/me")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# API endpoint tests — auth enabled via monkeypatch
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_register_and_login(monkeypatch):
    monkeypatch.setattr(jwt_mod, "AUTH_ENABLED", True)
    monkeypatch.setattr(_app_module, "AUTH_ENABLED", True)

    reg = client.post("/auth/register", json={"username": "alice", "password": "secret"})
    assert reg.status_code == 200
    assert reg.json()["username"] == "alice"

    login = client.post("/auth/token", data={"username": "alice", "password": "secret"})
    assert login.status_code == 200
    assert "access_token" in login.json()


@pytest.mark.unit
def test_me_with_token(monkeypatch):
    monkeypatch.setattr(jwt_mod, "AUTH_ENABLED", True)
    monkeypatch.setattr(_app_module, "AUTH_ENABLED", True)

    client.post("/auth/register", json={"username": "bob", "password": "hunter2"})
    login = client.post("/auth/token", data={"username": "bob", "password": "hunter2"})
    token = login.json()["access_token"]

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["username"] == "bob"


@pytest.mark.unit
def test_register_duplicate(monkeypatch):
    monkeypatch.setattr(jwt_mod, "AUTH_ENABLED", True)
    monkeypatch.setattr(_app_module, "AUTH_ENABLED", True)

    client.post("/auth/register", json={"username": "carol", "password": "pw1"})
    dup = client.post("/auth/register", json={"username": "carol", "password": "pw2"})
    assert dup.status_code == 409
    assert "already exists" in dup.json()["detail"].lower()
