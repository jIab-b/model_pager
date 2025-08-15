from __future__ import annotations

_STATE = None

def set_state(state):
    global _STATE
    _STATE = state

def get_state():
    return _STATE


