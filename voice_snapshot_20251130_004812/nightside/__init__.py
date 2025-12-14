"""
Nightside - The Metacognitive Layer for Astraeus

The quiet control room that watches everything the pilot + co-pilot do,
and occasionally taps them on the shoulder with patterns and contradictions.

Components:
- SilentChoir: Thread-safe thought logging to JSONL
- WellOfReflections: Theme extraction with time-decay
- ParadoxEngine: Diversity-aware reranking for RAG

Usage:
    from nightside import choir, reflect, paradox_rerank

    # Log a thought
    choir.record(topic="physics", content="What is dark matter?")

    # Get current themes
    themes = reflect(limit=10)

    # Rerank RAG hits for diversity
    reranked = paradox_rerank(query, hits)
"""

from .silent_choir import SilentChoir, ThoughtEvent, choir
from .well_of_reflections import (
    WellOfReflections,
    well_of_reflections,
    suggest_themes,
    get_session_context
)
from .paradox import ParadoxEngine, paradox_rerank, analyze_hits


def reflect(limit: int = 10, half_life_days: float = 7.0):
    """Get themes from the default choir with time decay"""
    events = choir.to_list(limit=500)
    return well_of_reflections(events, half_life_days=half_life_days, limit=limit)


def record(topic: str, content: str, **kwargs):
    """Record a thought to the default choir"""
    return choir.record(topic=topic, content=content, **kwargs)


__all__ = [
    'SilentChoir', 'ThoughtEvent', 'choir',
    'WellOfReflections', 'well_of_reflections', 'suggest_themes', 'get_session_context',
    'ParadoxEngine', 'paradox_rerank', 'analyze_hits',
    'record', 'reflect'
]

__version__ = "0.1.0"
