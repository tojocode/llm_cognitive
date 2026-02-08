# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Dict, List

from llm_core.types import Konzept, Verbindung


class WernickeMixin:
    # -------------------------
    # Init
    # -------------------------

    def _init_wernicke(self, lexikon_datei: str | None):
        self.lexikon_datei = lexikon_datei
        self.lexikon: Dict[str, str] = {}
        if self.lexikon_datei:
            self.lade_lexikon(self.lexikon_datei)

    # -------------------------
    # Lexikon
    # -------------------------

    def _norm_label(self, s: str) -> str:
        t = self._nfc(s).strip().lower()
        t = re.sub(r"[\.,;:!?()\[\]{}<>\"'`]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        for a in ["der ", "die ", "das ", "ein ", "eine ", "einen ", "einem ", "einer ", "den ", "dem ", "des "]:
            if t.startswith(a):
                t = t[len(a):].strip()
                break
        return t

    def lade_lexikon(self, datei: str):
        pfad = self._resolve_path(datei)
        if not os.path.exists(pfad):
            return
        try:
            with open(pfad, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        lex: Dict[str, str] = {}
        if isinstance(data, dict):
            for alias, cid in data.items():
                a = self._norm_label(str(alias))
                c = self._nfc(str(cid)).strip()
                if a and c:
                    lex[a] = c
        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                alias = item.get("alias")
                cid = item.get("id")
                a = self._norm_label(str(alias)) if alias else ""
                c = self._nfc(str(cid)).strip() if cid else ""
                if a and c:
                    lex[a] = c
        self.lexikon = lex

    def speichere_lexikon(self, datei: str | None = None):
        target = datei or self.lexikon_datei or "lexikon.json"
        pfad = self._resolve_path(target)
        with open(pfad, "w", encoding="utf-8") as f:
            json.dump(self.lexikon, f, ensure_ascii=False, indent=2)

    def lexikon_add(self, concept_id: str, alias: str) -> bool:
        cid = self._nfc(concept_id).strip()
        a = self._norm_label(alias)
        if not cid or not a:
            return False
        self.lexikon[a] = cid
        self._ensure_label(cid, alias)
        return True

    # -------------------------
    # Frage -> Cues
    # -------------------------

    def _tokenize(self, text: str) -> List[str]:
        txt = text.lower()
        txt = re.sub(r"[\.,;:!?()\[\]{}<>\"'`]", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt.split() if txt else []

    def _cue_set(self, frage: str) -> Dict[str, float]:
        tokens = self._tokenize(frage)
        tokset = set(tokens)
        stop = {
            "der","die","das","ein","eine","einen","einem","einer","ist","sind",
            "und","oder","zu","im","in","am","an","von","mit","für","für","den","dem","des",
            "was","wie","warum","wieso","weshalb"
        }
        tokset = {t for t in tokset if t not in stop}

        cues: Dict[str, float] = {}

        if self.lexikon:
            for alias, cid in self.lexikon.items():
                if not alias:
                    continue
                parts = [p for p in alias.split(" ") if p]
                if not parts:
                    continue
                if len(parts) == 1:
                    if parts[0] in tokset:
                        cues[cid] = max(cues.get(cid, 0.0), 0.93)
                        self._ensure_label(cid, alias)
                else:
                    if all(p in tokset for p in parts):
                        cues[cid] = max(cues.get(cid, 0.0), 0.90)
                        self._ensure_label(cid, alias)

        for kid, k in self.konzepte.items():
            kname = kid.lower()

            if kname in tokset:
                cues[kid] = max(cues.get(kid, 0.0), 0.95)
                continue

            if "_" in kname:
                parts = [p for p in kname.split("_") if p]
                if parts and all(p in tokset for p in parts):
                    cues[kid] = max(cues.get(kid, 0.0), 0.90)

            for lab in (k.labels or []):
                lab_norm = self._norm_label(lab)
                if not lab_norm:
                    continue
                lab_tokens = [t for t in lab_norm.split(" ") if t]
                if len(lab_tokens) == 1 and lab_tokens[0] in tokset:
                    cues[kid] = max(cues.get(kid, 0.0), 0.92)
                elif lab_tokens and all(t in tokset for t in lab_tokens):
                    cues[kid] = max(cues.get(kid, 0.0), 0.90)

            for ft in k.semantische_features:
                ftl = ft.lower()
                if ftl in tokset:
                    cues[kid] = max(cues.get(kid, 0.0), 0.75)

        return cues

    # -------------------------
    # Unknown / Stub (für Lern-Interface)
    # -------------------------

    def _make_concept_id(self, token: str) -> str:
        tok = self._nfc(token).strip()
        if not tok:
            return ""
        parts = [p for p in tok.split("_") if p]
        if len(parts) > 1:
            parts = [p[:1].upper() + p[1:] if p else p for p in parts]
            return "_".join(parts)
        return tok[:1].upper() + tok[1:]

    def _extract_topic_token(self, frage: str) -> str:
        tokens = self._tokenize(frage)
        if not tokens:
            return ""
        stop = {
            "der","die","das","ein","eine","einen","einem","einer","ist","sind",
            "und","oder","zu","im","in","am","an","von","mit","für","für","den","dem","des",
            "was","wie","warum","wieso","weshalb"
        }
        cand = [t for t in tokens if t not in stop and len(t) >= 3]
        return cand[-1] if cand else ""

    def _create_unknown_stub(self, frage: str) -> str:
        tok = self._extract_topic_token(frage)
        if not tok:
            return ""
        cid = self._make_concept_id(tok)
        if cid in self.konzepte:
            self._ensure_label(cid, tok)
            return cid
        self.konzepte[cid] = Konzept(id=cid, labels=[cid, tok], semantische_features=["unbekannt"])
        return cid

    # -------------------------
    # Lern-Interface: Text -> Nodes/Edges
    # -------------------------

    def import_text_file(self, pfad: str, target: str = "episodic") -> Dict[str, int]:
        p = self._resolve_path(pfad)
        with open(p, "r", encoding="utf-8") as f:
            txt = f.read()
        return self.import_text(txt, target=target, source=os.path.basename(p))

    def import_text(self, text: str, target: str = "episodic", source: str = "import") -> Dict[str, int]:
        target = (target or "episodic").strip().lower()
        if target not in {"semantic", "episodic"}:
            target = "episodic"

        nodes_before = len(self.konzepte)
        edges_before = sum(len(k.verbindungen) for k in self.konzepte.values()) + sum(len(v) for v in self.episodic_edges.values())

        extracted = self._extract_relations_from_text(text)

        nodes_added = 0
        edges_added = 0

        for rel in extracted:
            src = rel["src"]
            dst = rel["dst"]
            typ = rel["type"]
            w = rel["w"]
            src_label = rel.get("src_label") or src
            dst_label = rel.get("dst_label") or dst

            if src not in self.konzepte:
                self.konzepte[src] = Konzept(id=src, labels=[src_label, src], semantische_features=["gelernt"])
                nodes_added += 1
            if dst not in self.konzepte:
                self.konzepte[dst] = Konzept(id=dst, labels=[dst_label, dst], semantische_features=["gelernt"])
                nodes_added += 1
            self._ensure_label(src, src_label)
            self._ensure_label(dst, dst_label)

            if typ in {"ist", "klasse", "gehört_zu"}:
                feats = set(self.konzepte[src].semantische_features or [])
                if dst not in feats and len(feats) < 32:
                    self.konzepte[src].semantische_features.append(dst)

            if target == "semantic":
                exists = any(e.ziel == dst and e.typ == typ for e in self.konzepte[src].verbindungen)
                if not exists:
                    self.konzepte[src].verbindungen.append(Verbindung(ziel=dst, gewicht=w, typ=typ))
                    edges_added += 1
            else:
                self.episodic_edges.setdefault(src, [])
                exists = any(e.ziel == dst and e.typ == typ for e in self.episodic_edges[src])
                if not exists:
                    self.episodic_edges[src].append(Verbindung(ziel=dst, gewicht=w, typ=typ))
                    edges_added += 1

        if target == "episodic":
            stamp = datetime.now().strftime("%Y-%m-%d")
            tag = f"import:{source}:{stamp}"
            if "Import" not in self.konzepte:
                self.konzepte["Import"] = Konzept(id="Import", labels=["Import"], semantische_features=["Meta"])
                nodes_added += 1
            if tag not in self.konzepte["Import"].semantische_features:
                self.konzepte["Import"].semantische_features.append(tag)

        if getattr(self, "embed_store_on_import", False):
            try:
                self.embed_add_text(text, meta={"source": source, "target": target})
            except Exception:
                pass

        nodes_after = len(self.konzepte)
        edges_after = sum(len(k.verbindungen) for k in self.konzepte.values()) + sum(len(v) for v in self.episodic_edges.values())

        return {
            "nodes_before": nodes_before,
            "edges_before": edges_before,
            "nodes_after": nodes_after,
            "edges_after": edges_after,
            "nodes_added": nodes_added,
            "edges_added": edges_added,
        }

    def _extract_relations_from_text(self, text: str) -> List[Dict[str, object]]:
        txt = self._nfc(text)

        parts = re.split(r"[\n\r]+", txt)
        sentences: List[str] = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            sentences.extend([s.strip() for s in re.split(r"(?<=[\.!\?])\s+", p) if s.strip()])

        rels: List[Dict[str, object]] = []

        patterns = [
            (re.compile(r"^(.+?)\s+ist\s+ein(?:e|en|em|er)?\s+(.+?)\.?$", re.IGNORECASE), "ist", 0.95),
            (re.compile(r"^(.+?)\s+ist\s+(.+?)\.?$", re.IGNORECASE), "ist", 0.85),
            (re.compile(r"^(.+?)\s+gehört\s+zu\s+(.+?)\.?$", re.IGNORECASE), "gehört_zu", 0.93),
            (re.compile(r"^(.+?)\s+hat\s+(.+?)\.?$", re.IGNORECASE), "hat", 0.86),
            (re.compile(r"^(.+?)\s+enthält\s+(.+?)\.?$", re.IGNORECASE), "enthält", 0.86),
            (re.compile(r"^(.+?)\s+besteht\s+aus\s+(.+?)\.?$", re.IGNORECASE), "besteht_aus", 0.90),
            (re.compile(r"^(.+?)\s+braucht\s+(.+?)\.?$", re.IGNORECASE), "braucht", 0.86),
            (re.compile(r"^(.+?)\s+benötigt\s+(.+?)\.?$", re.IGNORECASE), "benötigt", 0.86),
            (re.compile(r"^(.+?)\s+ermöglicht\s+(.+?)\.?$", re.IGNORECASE), "ermöglicht", 0.88),
            (re.compile(r"^(.+?)\s+verursacht\s+(.+?)\.?$", re.IGNORECASE), "verursacht", 0.90),
            (re.compile(r"^(.+?)\s+lebt\s+in\s+(.+?)\.?$", re.IGNORECASE), "lebt_in", 0.83),
        ]

        for s in sentences:
            if len(s) < 5:
                continue
            s0 = s.strip("•*- \t\"'")
            for rx, typ, w0 in patterns:
                m = rx.match(s0)
                if not m:
                    continue
                subj_raw = m.group(1)
                obj_raw = m.group(2)

                src = self._phrase_to_concept_id(subj_raw)
                objs = self._split_object_phrases(obj_raw)

                for j, o in enumerate(objs):
                    dst = self._phrase_to_concept_id(o)
                    if not src or not dst or src == dst:
                        continue
                    rels.append({
                        "src": src,
                        "dst": dst,
                        "type": self._canon_type(typ),
                        "w": max(0.01, min(w0 - 0.02 * j, 0.99)),
                        "src_label": self._nfc(subj_raw).strip(),
                        "dst_label": self._nfc(o).strip(),
                    })
                break

        return rels

    def _phrase_to_concept_id(self, phrase: str) -> str:
        p = self._nfc(phrase)
        p = re.sub(r"[\(\)\[\]{}\"'`]", " ", p)
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            return ""
        p_norm = self._norm_label(p)
        if p_norm and p_norm in self.lexikon:
            cid = self.lexikon[p_norm]
            self._ensure_label(cid, p)
            return cid
        if p_norm:
            for kid, k in self.konzepte.items():
                for lab in (k.labels or []):
                    if self._norm_label(lab) == p_norm:
                        return kid
        p_low = p.lower()
        for a in ["der ", "die ", "das ", "ein ", "eine ", "einen ", "einem ", "einer ", "den ", "dem ", "des "]:
            if p_low.startswith(a):
                p = p[len(a):].strip()
                break
        p = p.split(",")[0].strip()
        words = [w for w in p.split(" ") if w][:3]
        if not words:
            return ""
        if len(words) == 1:
            return self._make_concept_id(words[0])
        return "_".join([self._make_concept_id(w) for w in words])

    def _split_object_phrases(self, obj: str) -> List[str]:
        o = self._nfc(obj).strip().strip(".")
        if not o:
            return []
        parts = []
        for chunk in re.split(r",|\s+und\s+", o):
            c = chunk.strip()
            if c:
                parts.append(c)
        return parts if parts else [o]
