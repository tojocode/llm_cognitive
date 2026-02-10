# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Dict, List, Optional, Tuple


def _normalize_text(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[\r\n\t]+", " ", t)
    t = re.sub(r"[^\wäöüß\- ]+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _fingerprint(text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    return h


class HashingEmbedder:
    def __init__(self, dim: int = 512, use_words: bool = True, use_char_ngrams: bool = True, ngram_min: int = 3, ngram_max: int = 5):
        self.dim = max(64, int(dim))
        self.use_words = use_words
        self.use_char_ngrams = use_char_ngrams
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max

    def _hash(self, token: str) -> int:
        h = hashlib.md5(token.encode("utf-8", errors="ignore")).hexdigest()
        return int(h, 16) % self.dim

    def _tokens(self, text: str) -> List[str]:
        t = _normalize_text(text)
        tokens: List[str] = []
        if self.use_words:
            tokens.extend([w for w in t.split(" ") if w])
        if self.use_char_ngrams:
            s = t.replace(" ", "")
            nmin, nmax = self.ngram_min, self.ngram_max
            if len(s) >= nmin:
                for n in range(nmin, min(nmax, len(s)) + 1):
                    for i in range(0, len(s) - n + 1):
                        tokens.append(s[i:i + n])
        return tokens

    def embed_sparse(self, text: str) -> Dict[int, float]:
        tokens = self._tokens(text)
        if not tokens:
            return {}
        counts: Dict[int, int] = {}
        for tok in tokens:
            idx = self._hash(tok)
            counts[idx] = counts.get(idx, 0) + 1
        # L2 normalize
        norm = sum(v * v for v in counts.values()) ** 0.5
        if norm == 0:
            return {}
        return {k: v / norm for k, v in counts.items()}


def cosine_sparse(a: Dict[int, float], b: Dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    # iterate smaller dict
    if len(a) > len(b):
        a, b = b, a
    s = 0.0
    for k, v in a.items():
        s += v * b.get(k, 0.0)
    return float(s)


class EmbeddingDB:
    def __init__(self, path: str, dim: int = 512):
        self.path = path
        self.dim = dim
        self.embedder = HashingEmbedder(dim=dim)
        self.items: List[Dict[str, object]] = []
        self._load()

    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return
        out: List[Dict[str, object]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            vec = it.get("vec") or {}
            vec_i: Dict[int, float] = {}
            if isinstance(vec, dict):
                for k, v in vec.items():
                    try:
                        vec_i[int(k)] = float(v)
                    except Exception:
                        continue
            out.append({
                "id": it.get("id"),
                "text": it.get("text"),
                "meta": it.get("meta") or {},
                "vec": vec_i,
            })
        self.items = out

    def save(self):
        if not self.path:
            return
        payload = {
            "format": "hashing-embed-v1",
            "dim": self.dim,
            "items": [
                {
                    "id": it.get("id"),
                    "text": it.get("text"),
                    "meta": it.get("meta") or {},
                    "vec": {str(k): float(v) for k, v in (it.get("vec") or {}).items()},
                }
                for it in self.items
            ],
        }
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def add(self, text: str, meta: Optional[Dict[str, object]] = None) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        norm = _normalize_text(t)
        fp = _fingerprint(norm)
        for it in self.items:
            if it.get("id") == fp:
                return False
        vec = self.embedder.embed_sparse(t)
        self.items.append({
            "id": fp,
            "text": t,
            "meta": meta or {},
            "vec": vec,
        })
        return True

    def query(self, text: str, top_k: int = 5, min_score: float = 0.2) -> List[Dict[str, object]]:
        t = (text or "").strip()
        if not t:
            return []
        q = self.embedder.embed_sparse(t)
        if not q:
            return []
        scored: List[Tuple[float, Dict[str, object]]] = []
        for it in self.items:
            score = cosine_sparse(q, it.get("vec") or {})
            if score >= min_score:
                scored.append((score, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, it in scored[: max(1, top_k)]:
            out.append({
                "score": score,
                "text": it.get("text") or "",
                "meta": it.get("meta") or {},
                "id": it.get("id"),
            })
        return out
