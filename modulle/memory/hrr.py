"""Holographic Reduced Representations (HRR) memory for ModuLLe.

Pure-python implementation (no numpy dependency) of Plate's HRR:
- Vector elements: real numbers; similarity = normalized dot (cosine).
- Binding is circular convolution; unbinding via correlation.
- Items are encoded with fractional power encoding (FPE) so structured data
  (role-filler pairs like ``topic=Python``) can be stored and probed.

Memory store persists entries as JSON at ``memory.json`` with the vectors
stored as lists of floats (dimension configurable; 2048 default keeps
dot-products cheap while remaining noise-tolerant).

Example:
    >>> from modulle.memory import HRRMemoryStore
    >>> mem = HRRMemoryStore(path='~/.disenchanted/memory.json')
    >>> mem.store("User prefers dark themes.", topics=['preferences'])
    >>> hits = mem.recall("what theme does the user like?", top_k=3)
"""

import json
import math
import random
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from modulle.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Vector primitives (lists of floats; index math by Python)
# ---------------------------------------------------------------------------
def hrr_zeros(n: int) -> List[float]:
    return [0.0] * n


def hrr_random(n: int, rng: random.Random) -> List[float]:
    """Random Gaussian vector (approx. unit norm for large n)."""
    return [rng.gauss(0.0, 1.0 / math.sqrt(n)) for _ in range(n)]


def hrr_unit(v: List[float]) -> List[float]:
    """Unit-normalize a vector (safe on zero vectors)."""
    n = math.sqrt(sum(x * x for x in v))
    if n < 1e-12:
        return list(v)
    return [x / n for x in v]


def hrr_convolve(a: List[float], b: List[float]) -> List[float]:
    """Circular convolution — the HRR binding operator, O(n^2)."""
    n = len(a)
    # Direct circular convolution
    out = [0.0] * n
    for i in range(n):
        ai = a[i]
        if ai == 0.0:
            continue
        for j in range(n):
            out[(i + j) % n] += ai * b[j]
    return out


def hrr_correlate(a: List[float], c: List[float]) -> List[float]:
    """Circular correlation (inverse of convolution) — unbinding."""
    n = len(a)
    out = [0.0] * n
    for i in range(n):
        ai = a[i]
        if ai == 0.0:
            continue
        for j in range(n):
            out[(i - j) % n] += ai * c[j]
    return out


def hrr_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity of two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * y for x, y in zip(b, b)))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def hrr_base(n: int, seed: int) -> List[float]:
    """Deterministic random base vector for a given seed."""
    rng = random.Random(seed * 1000003 + 17)
    return hrr_random(n, rng)


def hrr_encode_pair(role: List[float], filler: List[float], n: int) -> List[float]:
    """Encode a role/filler pair as role ⊛ filler (kept for structured HRR use)."""
    return hrr_convolve(role, filler)


def hrr_sum(vectors: List[List[float]]) -> List[float]:
    """Superposition (element-wise sum) of vectors."""
    if not vectors:
        return []
    n = len(vectors[0])
    return [sum(v[i] for v in vectors) for i in range(n)]


def text_vector(text: str, n: int, seed_base: int = 12345) -> List[float]:
    """Deterministic bag-of-words HRR vector for arbitrary text.

    Each word hashes to a deterministic unit-norm base vector; the text
    vector is the normalized superposition of its word vectors. Deterministic
    across processes (stable hash, not Python's salted hash()); similarity
    tracks word overlap.
    """
    acc = hrr_zeros(n)
    count = 0
    for w in re.split(r"\W+", text.lower()):
        if not w:
            continue
        seed = (_stable_hash(w) ^ (seed_base * 2654435761)) & 0xFFFFFFFF
        acc = hrr_sum([acc, hrr_base(n, seed)])
        count += 1
    if count == 0:
        return hrr_unit(hrr_base(n, _stable_hash(text) ^ seed_base))
    return hrr_unit(acc)


def _stable_hash(s: str) -> int:
    """Deterministic string hash (independent of PYTHONHASHSEED)."""
    h = 2166136261
    for ch in s:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h


class HRRMemoryStore:
    """Episodic memory store with HRR-based associative recall.

    Entries store text plus vector; recall ranks by cosine similarity of the
    query vector against entry vectors.

    Args:
        path: JSON file for persistence (``~/.disenchanted/memory.json``).
        dim: HRR vector dimension (even number recommended).
        max_entries: Oldest entries evicted beyond this (0 = unlimited).
    """

    def __init__(self, path: Optional[str] = None, dim: int = 2048, max_entries: int = 0):
        self.dim = dim if dim % 2 == 0 else dim + 1
        self.max_entries = max_entries
        self.path = (
            Path(path).expanduser() if path else (Path.home() / ".disenchanted" / "memory.json")
        )
        self.entries: List[Dict[str, Any]] = []
        self._load()

    # -- persistence ----------------------------------------------------------
    def _load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = [e for e in data.get("entries", []) if self._valid_entry(e)]
            if len(self.entries) > self.max_entries > 0:
                self.entries = self.entries[-self.max_entries :]
        except Exception as e:
            logger.error(f"Failed to load HRR memory {self.path}: {e}")
            self.entries = []

    @staticmethod
    def _valid_entry(e) -> bool:
        return (
            isinstance(e, dict)
            and isinstance(e.get("vector"), list)
            and len(e.get("vector", [])) > 0
        )

    def save(self):
        """Persist memory to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.write_text(
                json.dumps({"dim": self.dim, "entries": self.entries}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"Failed to save HRR memory: {e}")
            raise

    # -- operations -------------------------------------------------------------
    def store(
        self, text: str, topics: Optional[List[str]] = None, timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """Add an episodic memory entry; returns the stored entry."""
        vec = text_vector(f"{' '.join(topics or [])} {text}", self.dim)
        entry = {
            "id": uuid.uuid4().hex[:12],
            "text": text,
            "topics": list(topics or []),
            "timestamp": timestamp if timestamp is not None else time.time(),
            "vector": vec,
        }
        self.entries.append(entry)
        if self.max_entries and len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]
        return entry

    def recall(
        self, query: str, top_k: int = 3, min_similarity: float = 0.05, topic: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find entries most similar to the query text.

        Returns up to ``top_k`` entries (without vectors) above
        ``min_similarity``, optionally filtered by topic.
        """
        if not self.entries:
            return []
        qv = text_vector(query, self.dim)
        scored = []
        for e in self.entries:
            if topic is not None and topic not in e.get("topics", []):
                continue
            sim = hrr_similarity(qv, e["vector"])
            if sim >= min_similarity:
                scored.append((sim, e))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            {
                "id": e["id"],
                "text": e["text"],
                "topics": e["topics"],
                "timestamp": e["timestamp"],
                "similarity": round(sim, 4),
            }
            for sim, e in scored[:top_k]
        ]

    def as_context_note(self, query: str, top_k: int = 3) -> Optional[str]:
        """Format recall results as an injectable context note (or None)."""
        hits = self.recall(query, top_k=top_k)
        if not hits:
            return None
        lines = ["Relevant memories:"]
        for h in hits:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(h["timestamp"]))
            lines.append(f"- ({ts}) {h['text']}")
        return "\n".join(lines)

    def edit(
        self, entry_id: str, new_text: Optional[str] = None, topics: Optional[List[str]] = None
    ) -> bool:
        """Edit an entry's text/topics; re-encodes the vector."""
        for e in self.entries:
            if e["id"] == entry_id:
                if new_text is not None:
                    e["text"] = new_text
                if topics is not None:
                    e["topics"] = list(topics)
                e["vector"] = text_vector(f"{' '.join(e['topics'])} {e['text']}", self.dim)
                return True
        return False

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by id; True if found."""
        before = len(self.entries)
        self.entries = [e for e in self.entries if e["id"] != entry_id]
        return len(self.entries) < before

    def clear(self):
        """Remove all entries."""
        self.entries = []

    def list_entries(self, include_vectors: bool = False) -> List[Dict[str, Any]]:
        """List entries (vectors excluded by default)."""
        return [
            {k: v for k, v in e.items() if include_vectors or k != "vector"} for e in self.entries
        ]
