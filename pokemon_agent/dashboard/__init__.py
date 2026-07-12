"""Hermes Plays Pokémon — Dashboard package."""

from .history import EventLogger
from .mount import mount_dashboard

__all__ = ["mount_dashboard", "EventLogger"]
