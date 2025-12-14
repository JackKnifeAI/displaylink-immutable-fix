"""
Well of Reflections: theme extraction with time-decay.
Safe: no claims of prediction; just pattern surfacing.

The well watches what flows through SilentChoir and surfaces
what the mind keeps circling back to.
"""
from __future__ import annotations
import math
import time
from collections import defaultdict
from typing import List, Dict, Any, Optional
from .silent_choir import choir, ThoughtEvent


class WellOfReflections:
    """
    Extracts themes from thought events with time-decay weighting.

    Half-life controls how fast old thoughts fade:
    - 7 days: good for journaling/reflection
    - 1 day: good for session-based agents
    - 1 hour: good for tight agent loops
    """

    def __init__(self, half_life_days: float = 7.0):
        self.half_life_seconds = half_life_days * 24 * 3600

    def compute_weight(self, ts: float, now: Optional[float] = None) -> float:
        """Compute time-decay weight: 2^(-age/half_life)"""
        now = now or time.time()
        age_seconds = max(0, now - ts)
        return math.pow(2, -age_seconds / self.half_life_seconds)

    def extract_themes(
        self,
        events: List[Dict[str, Any]],
        limit: int = 10,
        min_count: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Extract themes from events with time-decay weighting.

        Returns list of:
        {
            "theme": str,
            "score": float (time-weighted),
            "count": int (raw count),
            "recent": bool (any in last 24h)
        }
        """
        now = time.time()
        day_ago = now - 86400

        themes: Dict[str, Dict] = defaultdict(lambda: {
            "score": 0.0,
            "count": 0,
            "recent": False
        })

        for evt in events:
            # Extract theme from topic
            topic = evt.get("topic", "").lower().strip()
            if not topic:
                continue

            ts = evt.get("ts", now)
            weight = self.compute_weight(ts, now)

            themes[topic]["score"] += weight
            themes[topic]["count"] += 1
            if ts > day_ago:
                themes[topic]["recent"] = True

            # Also extract from tags
            for tag in evt.get("tags", []):
                tag = tag.lower().strip()
                if tag and tag != topic:
                    themes[tag]["score"] += weight * 0.5  # Tags worth half
                    themes[tag]["count"] += 1
                    if ts > day_ago:
                        themes[tag]["recent"] = True

        # Filter and rank
        results = [
            {"theme": k, **v}
            for k, v in themes.items()
            if v["count"] >= min_count
        ]
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:limit]


def well_of_reflections(
    events: List[Dict[str, Any]],
    half_life_days: float = 7.0,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Convenience function for theme extraction"""
    well = WellOfReflections(half_life_days=half_life_days)
    return well.extract_themes(events, limit=limit)


def suggest_themes(limit: int = 5, half_life_days: float = 7.0) -> List[Dict]:
    """Legacy API - get themes from default choir"""
    events = choir.to_list(limit=500)
    return well_of_reflections(events, half_life_days=half_life_days, limit=limit)


def get_session_context(limit: int = 3, half_life_hours: float = 2.0) -> str:
    """
    Get a context string for injection into system prompts.

    Returns something like:
    "Recent focus areas: python programming, neural networks, debugging"
    """
    events = choir.to_list(limit=200)
    themes = well_of_reflections(
        events,
        half_life_days=half_life_hours / 24,
        limit=limit
    )

    if not themes:
        return ""

    theme_names = [t["theme"] for t in themes if t["recent"]]
    if not theme_names:
        theme_names = [t["theme"] for t in themes[:limit]]

    if not theme_names:
        return ""

    return f"Recent focus areas: {', '.join(theme_names)}"
