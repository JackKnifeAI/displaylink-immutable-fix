"""
Silent Choir: in-memory + optional persistent log of ThoughtEvents.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from config import OFFLINE_ROOT  # type: ignore
except Exception:
    OFFLINE_ROOT = Path(os.environ.get("OFFLINE_AI_ROOT", "/T7_OFFLINE_AI")).resolve()

NIGHTSIDE_DIR = OFFLINE_ROOT / "07_nightside"
NIGHTSIDE_LOG = NIGHTSIDE_DIR / "thought_log.jsonl"


@dataclass
class ThoughtEvent:
    id: int
    kind: str
    topic: str
    text: str
    ts: float
    tags: List[str]
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SilentChoir:
    """
    Lightweight event log with optional JSONL persistence.
    """

    def __init__(self, log_path: Optional[Path] = None):
        self.events: List[ThoughtEvent] = []
        self._next_id = 1
        self._lock = threading.Lock()
        self.log_path = log_path
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def _load_from_disk(self, max_events: int = 2000) -> None:
        if not self.log_path or not self.log_path.exists():
            return
        try:
            with self.log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    ev = ThoughtEvent(
                        id=int(obj.get("id", 0) or 0),
                        kind=str(obj.get("kind", "")),
                        topic=str(obj.get("topic", "")),
                        text=str(obj.get("text", "")),
                        ts=float(obj.get("ts", time.time())),
                        tags=list(obj.get("tags", [])),
                        meta=dict(obj.get("meta", {})),
                    )
                    self.events.append(ev)
            if len(self.events) > max_events:
                self.events = self.events[-max_events:]
            if self.events:
                self._next_id = max(ev.id for ev in self.events) + 1
        except Exception:
            # Corruption shouldn't crash the system; ignore and start fresh
            self.events = []
            self._next_id = 1

    def _append_to_disk(self, ev: ThoughtEvent) -> None:
        if not self.log_path:
            return
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")
        except Exception:
            pass

    def record(
        self,
        topic: str,
        content: str,
        *,
        kind: Optional[str] = None,
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        tags = tags or []
        meta = meta or {}
        if kind is None:
            kind = topic
        with self._lock:
            ev = ThoughtEvent(
                id=self._next_id,
                kind=str(kind),
                topic=str(topic),
                text=str(content),
                ts=float(time.time()),
                tags=list(tags),
                meta=dict(meta),
            )
            self.events.append(ev)
            self._next_id += 1
            # Keep bounded
            if len(self.events) > 5000:
                self.events = self.events[-2000:]
        self._append_to_disk(ev)
        return ev

    def to_list(self, limit: int = 256) -> List[Dict[str, Any]]:
        with self._lock:
            return [e.to_dict() for e in self.events[-limit:]]

    def clear(self, clear_disk: bool = False):
        with self._lock:
            self.events = []
            self._next_id = 1
        if clear_disk and self.log_path and self.log_path.exists():
            try:
                self.log_path.unlink()
            except Exception:
                pass

    # Convenience
    def record_query(self, text: str, meta: Optional[Dict[str, Any]] = None):
        return self.record("qa_query", text, kind="query", tags=["qa", "query"], meta=meta)

    def record_answer(self, text: str, meta: Optional[Dict[str, Any]] = None):
        return self.record("qa_answer", text, kind="answer", tags=["qa", "answer"], meta=meta)


choir = SilentChoir(log_path=NIGHTSIDE_LOG)
