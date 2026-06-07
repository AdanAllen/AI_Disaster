import secrets

from app import RESIDENT_STATE_CACHE, RESIDENT_STATE_TTL_SECONDS, RESULT_TOKEN_SESSION_KEY


def set_test_resident_state(client, state):
    """Seed ephemeral resident state without putting private values in the cookie."""
    token = secrets.token_urlsafe(24)
    RESIDENT_STATE_CACHE.set(token, dict(state), timeout=RESIDENT_STATE_TTL_SECONDS)
    with client.session_transaction() as saved:
        saved.clear()
        saved[RESULT_TOKEN_SESSION_KEY] = token
    return token
