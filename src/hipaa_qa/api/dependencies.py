"""FastAPI dependency injection functions."""

from .state import AppState, get_app_state as _get_app_state


def get_app_state() -> AppState:
    """Dependency to get the application state."""
    return _get_app_state()