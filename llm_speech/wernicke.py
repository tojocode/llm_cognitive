# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Dict, List

from llm_core.types import Konzept, Verbindung

ARTICLES = [
    "der ",
    "die ",
    "das ",
    "ein ",
    "eine ",
    "einen ",
    "einem ",
    "einer ",
    "den ",
    "dem ",
    "des ",
]
JUNK_IDS = {
    "es",
    "er",
    "sie",
    "man",
    "dies",
    "diese",
    "dieser",
    "dieses",
    "aber",
    "auch",
    "denn",
    "dann",
    "daher",
    "deshalb",
    "daneben",
    "außerdem",
    "ausserdem",
    "hierbei",
    "dabei",
    "somit",
    "jedoch",
    "hat",
    "ist",
    "sind",
    "wird",
    "werden",
    "verb",
}
JUNK_TOKENS = {
    "der",
    "die",
    "das",
    "ein",
    "eine",
    "einen",
    "einem",
    "einer",
    "den",
    "dem",
    "des",
    "und",
    "oder",
    "zu",
    "im",
    "in",
    "am",
    "an",
    "von",
    "mit",
    "für",
    "fuer",
    "auch",
    "aber",
    "sowie",
    "manche",
    "einige",
    "viele",
    "mehr",
    "weniger",
    "andere",
    "anderen",
    "dies",
    "diese",
    "dieser",
    "dieses",
    "deshalb",
    "daher",
    "daneben",
    "außerdem",
    "ausserdem",
    "hierbei",
    "dabei",
    "somit",
    "jedoch",
    "begriff",
    "teil",
    "gegenstand",
    "definitionsgeschichte",
}
VERB_TOKENS = {
    "ist",
    "sind",
    "war",
    "wird",
    "werden",
    "hat",
    "haben",
    "behandelt",
    "zeigt",
    "gibt",
    "gibt_es",
    "heißt",
    "heisst",
    "bedeutet",
    "entsteht",
    "verursacht",
    "führt",
    "fuehrt",
    "besteht",
    "bestehen",
    "setzt",
    "umfasst",
    "enthält",
    "enthaelt",
    "lebt",
    "kommt",
}


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
        for a in ARTICLES:
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

    def _normalize_import_text(self, text: str) -> str:
        if not text:
            return ""
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        # join hyphenated line breaks
        t = re.sub(r"(?<=[A-Za-zÄÖÜäöüß])-\n(?=[A-Za-zÄÖÜäöüß])", "", t)
        # join broken words across line breaks
        t = re.sub(r"(?<=[A-Za-zÄÖÜäöüß])\n(?=[a-zäöüß])", "", t)
        # fix common OCR/line-break artifacts seen in imports
        t = re.sub(r"\bVerb und\b", "Verbund", t)
        t = t.replace("Grundlagenft", "Grundlagen- und")
        return t

    def _is_junk_concept_id(self, cid: str) -> bool:
        if not cid:
            return True
        low = cid.lower()
        if low in JUNK_IDS:
            return True
        if cid.isdigit():
            return True
        if len(cid) <= 2 and not cid.isupper():
            return True
        tokens = [t for t in cid.split("_") if t]
        if not tokens:
            return True
        junk_count = 0
        alpha_count = 0
        has_verb = False
        for t in tokens:
            t_low = t.lower()
            if any(ch.isalpha() for ch in t_low):
                alpha_count += 1
            if t_low in VERB_TOKENS:
                has_verb = True
                junk_count += 1
                continue
            if t_low in JUNK_TOKENS:
                junk_count += 1
            if t_low.isdigit():
                junk_count += 1
            if re.fullmatch(r"\d+(\.\d+)?", t_low):
                junk_count += 1
        if alpha_count == 0:
            return True
        if has_verb:
            return True
        if any(t.lower() in {"oder", "und"} for t in tokens):
            return True
        return (junk_count / len(tokens)) >= 0.4

    def _cue_set(self, frage: str) -> Dict[str, float]:
        tokens = self._tokenize(frage)
        tokset = set(tokens)
        token_text = " " + " ".join(tokens) + " " if tokens else " "
        stop = {
            "der","die","das","ein","eine","einen","einem","einer","ist","sind",
            "und","oder","zu","im","in","am","an","von","mit","für","für","den","dem","des",
            "hat","haben","besteht","bestehen","lebt","gibt",
            "was","wie","warum","wieso","weshalb","woraus","womit","wodurch","wo","wann",
            "wer","wen","wem","wessen","welche","welcher","welches","welchen","welchem",
            "auch","aber","denn","dann","daher","deshalb","daneben",
            "außerdem","ausserdem","hierbei","dabei","somit","jedoch"
        }
        tokset = {t for t in tokset if t not in stop}

        cues: Dict[str, float] = {}

        if self.lexikon:
            for alias, cid in self.lexikon.items():
                if not alias:
                    continue
                if self._is_junk_concept_id(cid):
                    continue
                parts = [p for p in alias.split(" ") if p]
                if not parts:
                    continue
                if len(parts) == 1:
                    if parts[0] in tokset:
                        cues[cid] = max(cues.get(cid, 0.0), 0.93)
                        self._ensure_label(cid, alias)
                else:
                    phrase = " ".join(parts)
                    if f" {phrase} " in token_text:
                        cues[cid] = max(cues.get(cid, 0.0), 0.97)
                        self._ensure_label(cid, alias)
                    elif all(p in tokset for p in parts):
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
            "hat","haben","besteht","bestehen","lebt","gibt",
            "was","wie","warum","wieso","weshalb","woraus","womit","wodurch","wo","wann",
            "wer","wen","wem","wessen","welche","welcher","welches","welchen","welchem",
            "auch","aber","denn","dann","daher","deshalb","daneben",
            "außerdem","ausserdem","hierbei","dabei","somit","jedoch"
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

    def import_text(
        self,
        text: str,
        target: str = "episodic",
        source: str = "import",
    ) -> Dict[str, int]:
        text = self._normalize_import_text(text)
        target = (target or "episodic").strip().lower()
        if target not in {"semantic", "episodic"}:
            target = "episodic"

        nodes_before = len(self.konzepte)
        edges_before = sum(len(k.verbindungen) for k in self.konzepte.values()) + sum(
            len(v) for v in self.episodic_edges.values()
        )

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
                self.konzepte[src] = Konzept(
                    id=src,
                    labels=[src_label, src],
                    semantische_features=["gelernt"],
                )
                nodes_added += 1
            if dst not in self.konzepte:
                self.konzepte[dst] = Konzept(
                    id=dst,
                    labels=[dst_label, dst],
                    semantische_features=["gelernt"],
                )
                nodes_added += 1
            self._ensure_label(src, src_label)
            self._ensure_label(dst, dst_label)
            # Kurze Aliase für bessere Cues (z. B. "Axolotl")
            src_alias = self._head_alias(src_label)
            if src_alias:
                self._ensure_label(src, src_alias)
            dst_alias = self._head_alias(dst_label)
            if dst_alias:
                self._ensure_label(dst, dst_alias)

            if typ in {"ist", "klasse", "gehört_zu"}:
                feats = set(self.konzepte[src].semantische_features or [])
                if dst not in feats and len(feats) < 32:
                    self.konzepte[src].semantische_features.append(dst)

            if target == "semantic":
                exists = any(
                    e.ziel == dst and e.typ == typ for e in self.konzepte[src].verbindungen
                )
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
                self.konzepte["Import"] = Konzept(
                    id="Import",
                    labels=["Import"],
                    semantische_features=["Meta"],
                )
                nodes_added += 1
            if tag not in self.konzepte["Import"].semantische_features:
                self.konzepte["Import"].semantische_features.append(tag)

        if getattr(self, "embed_store_on_import", False):
            try:
                self.embed_add_text(text, meta={"source": source, "target": target})
            except Exception:
                pass

        nodes_after = len(self.konzepte)
        edges_after = sum(len(k.verbindungen) for k in self.konzepte.values()) + sum(
            len(v) for v in self.episodic_edges.values()
        )

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
            (
                re.compile(r"^Als\s+(.+?)\s+bezeichnet\s+man\s+(.+?)\.?$", re.IGNORECASE),
                "ist",
                0.92,
                False,
            ),
            (
                re.compile(r"^(.+?)\s+wird\s+als\s+(.+?)\s+bezeichnet\.?$", re.IGNORECASE),
                "ist",
                0.90,
                False,
            ),
            (
                re.compile(r"^(.+?)\s+ist\s+die\s+bezeichnung\s+für\s+(.+?)\.?$", re.IGNORECASE),
                "ist",
                0.90,
                False,
            ),
            (
                re.compile(r"^(.+?)\s+bezeichnet\s+(?:man\s+)?(.+?)\.?$", re.IGNORECASE),
                "ist",
                0.90,
                False,
            ),
            (
                re.compile(r"^(.+?)\s+setzt\s+sich\s+zusammen\s+aus\s+(.+?)\.?$", re.IGNORECASE),
                "besteht_aus",
                0.90,
                True,
            ),
            (
                re.compile(r"^(.+?)\s+umfasst\s+(.+?)\.?$", re.IGNORECASE),
                "enthält",
                0.86,
                True,
            ),
            (
                re.compile(r"^(.+?)\s+enthält\s+(.+?)\.?$", re.IGNORECASE),
                "enthält",
                0.86,
                True,
            ),
            (
                re.compile(r"^(.+?)\s+führt\s+zu\s+(.+?)\.?$", re.IGNORECASE),
                "verursacht",
                0.88,
                True,
            ),
            (
                re.compile(r"^(.+?)\s+wird\s+durch\s+(.+?)\s+verursacht\.?$", re.IGNORECASE),
                "verursacht_durch",
                0.90,
                True,
            ),
            (
                re.compile(r"^(.+?)\s+ist\s+durch\s+(.+?)\s+bedingt\.?$", re.IGNORECASE),
                "verursacht_durch",
                0.88,
                True,
            ),
            (
                re.compile(r"^(.+?)\s+entsteht\s+durch\s+(.+?)\.?$", re.IGNORECASE),
                "verursacht_durch",
                0.86,
                True,
            ),
            (
                re.compile(r"^(.+?)\s+entsteht\s+aus\s+(.+?)\.?$", re.IGNORECASE),
                "besteht_aus",
                0.86,
                True,
            ),
            (
                re.compile(r"^(.+?)\s+ist\s+ein(?:e|en|em|er)?\s+(.+?)\.?$", re.IGNORECASE),
                "ist",
                0.95,
                False,
            ),
            (
                re.compile(r"^(.+?)\s+ist\s+(.+?)\.?$", re.IGNORECASE),
                "ist",
                0.85,
                False,
            ),
            (
                re.compile(r"^(.+?)\s+gehört\s+zu\s+(.+?)\.?$", re.IGNORECASE),
                "gehört_zu",
                0.93,
                False,
            ),
            (
                re.compile(r"^(.+?)\s+hat\s+(.+?)\.?$", re.IGNORECASE),
                "hat",
                0.86,
                True,
            ),
            (
                re.compile(r"^(.+?)\s+besteht\s+aus\s+(.+?)\.?$", re.IGNORECASE),
                "besteht_aus",
                0.90,
                True,
            ),
            (
                re.compile(r"^(.+?)\s+braucht\s+(.+?)\.?$", re.IGNORECASE),
                "braucht",
                0.86,
                True,
            ),
            (
                re.compile(r"^(.+?)\s+benötigt\s+(.+?)\.?$", re.IGNORECASE),
                "benötigt",
                0.86,
                True,
            ),
            (
                re.compile(r"^(.+?)\s+ermöglicht\s+(.+?)\.?$", re.IGNORECASE),
                "ermöglicht",
                0.88,
                True,
            ),
            (
                re.compile(r"^(.+?)\s+verursacht\s+(.+?)\.?$", re.IGNORECASE),
                "verursacht",
                0.90,
                True,
            ),
            (
                re.compile(r"^(.+?)\s+lebt\s+in\s+(.+?)\.?$", re.IGNORECASE),
                "lebt_in",
                0.83,
                False,
            ),
            (
                re.compile(r"^(.+?)\s+kommt\s+in\s+(.+?)\s+vor\.?$", re.IGNORECASE),
                "lebt_in",
                0.80,
                False,
            ),
        ]

        for s in sentences:
            if len(s) < 5:
                continue
            s0 = s.strip("•*- \t\"'")
            live_rx = re.compile(
                r"^(.+?)\s+ist\s+ein\s+in\s+(.+?)\s+lebend(?:e[rn]?|)\b",
                re.IGNORECASE,
            )
            m_live = live_rx.match(s0)
            if m_live:
                src = self._phrase_to_concept_id(m_live.group(1))
                place_tokens = [t for t in re.split(r"\s+", m_live.group(2)) if t][:3]
                place = " ".join(place_tokens)
                dst = self._phrase_to_concept_id(place)
                if (
                    src
                    and dst
                    and not self._is_junk_concept_id(src)
                    and not self._is_junk_concept_id(dst)
                ):
                    rels.append(
                        {
                            "src": src,
                            "dst": dst,
                            "type": "lebt_in",
                            "w": 0.82,
                            "src_label": self._nfc(m_live.group(1)).strip(),
                            "dst_label": self._nfc(place).strip(),
                        }
                    )
            for rx, typ, w0, split_conj in patterns:
                m = rx.match(s0)
                if not m:
                    continue
                subj_raw = m.group(1)
                obj_raw = m.group(2)

                src = self._phrase_to_concept_id(subj_raw)
                objs = self._split_object_phrases(obj_raw, split_conjunctions=split_conj)

                for j, o in enumerate(objs):
                    o_head = self._object_head_phrase(o) if typ in {"ist", "gehört_zu"} else o
                    dst = self._phrase_to_concept_id(o_head)
                    if (
                        not src
                        or not dst
                        or src == dst
                        or self._is_junk_concept_id(src)
                        or self._is_junk_concept_id(dst)
                    ):
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
            return cid if not self._is_junk_concept_id(cid) else ""
        if p_norm:
            for kid, k in self.konzepte.items():
                for lab in (k.labels or []):
                    if self._norm_label(lab) == p_norm:
                        return kid if not self._is_junk_concept_id(kid) else ""
        p_low = p.lower()
        for a in ARTICLES:
            if p_low.startswith(a):
                p = p[len(a):].strip()
                break
        p = p.split(",")[0].strip()
        words = [w for w in p.split(" ") if w][:3]
        if not words:
            return ""
        if len(words) == 1:
            cid = self._make_concept_id(words[0])
        else:
            cid = "_".join([self._make_concept_id(w) for w in words])
        return cid if not self._is_junk_concept_id(cid) else ""

    def _split_object_phrases(self, obj: str, split_conjunctions: bool = True) -> List[str]:
        o = self._nfc(obj).strip().strip(".")
        if not o:
            return []
        parts = []
        if split_conjunctions:
            chunks = re.split(r",|\s+und\s+", o)
        else:
            chunks = [o]
        for chunk in chunks:
            c = chunk.strip()
            if c:
                parts.append(c)
        return parts if parts else [o]

    def _cut_at_keywords(self, text: str, keywords: List[str]) -> str:
        if not text:
            return text
        low = text.lower()
        best = None
        for kw in keywords:
            k = kw.lower()
            idx = low.find(k)
            if idx > 0 and (best is None or idx < best):
                best = idx
        if best is None:
            return text
        return text[:best].strip()

    def _object_head_phrase(self, obj: str) -> str:
        """
        Reduziert Objekt-Phrasen auf ein plausibles Kopfwort
        (z. B. "Handlungsprinzip bei der Nutzung ..." -> "Handlungsprinzip").
        """
        if not obj:
            return ""
        p = self._nfc(obj)
        p = re.sub(r"\([^)]*\)", " ", p)
        p = re.sub(r"[\"'“”„]", " ", p)
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            return ""

        if "," in p:
            p = p.split(",", 1)[0].strip()

        p = self._cut_at_keywords(p, [
            " bei ", " in ", " mit ", " für ", " fuer ", " von ", " durch ", " aus ", " auf ",
            " unter ", " über ", " ueber ", " zwischen ", " ohne ", " als ", " seit ", " nach ",
            " während ", " waehrend ",
        ])
        p = self._cut_at_keywords(p, [" des ", " der ", " dem ", " den ", " eines ", " einer "])

        p_low = p.lower()
        for a in ARTICLES:
            if p_low.startswith(a):
                p = p[len(a):].strip()
                break

        if not p:
            return ""

        stop = {
            "der", "die", "das", "ein", "eine", "einen", "einem", "einer", "den", "dem", "des",
            "und", "oder", "zu", "im", "in", "am", "an", "von", "mit", "für", "fuer",
            "was", "wie", "warum", "wieso", "weshalb", "diese", "dieser", "dieses", "dabei",
            "bei", "nach", "vor", "auf", "unter", "über", "ueber", "zwischen", "ohne", "als",
            "seit", "aus", "durch", "wegen", "gegen", "während", "waehrend",
        }

        nouns = re.findall(r"[A-ZÄÖÜ][a-zäöüß\-]+", p)
        if nouns:
            nouns = [n for n in nouns if n.lower() not in stop]
            if nouns:
                return nouns[-1]

        for t in re.split(r"\s+", p):
            t0 = re.sub(r"[^\wäöüß\-]+", "", t, flags=re.IGNORECASE)
            if len(t0) >= 3 and t0.lower() not in stop:
                return t0
        return ""

    def _head_alias(self, phrase: str) -> str:
        """
        Extrahiert ein kurzes Alias (Headword) aus einer Phrase, um Cues
        bei Einzelwort-Fragen zu ermöglichen (z. B. "Axolotl").
        """
        if not phrase:
            return ""
        p = self._nfc(phrase)
        p = re.sub(r"\([^)]*\)", " ", p)
        p = re.sub(r"[,;:]", " ", p)
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            return ""

        p_low = p.lower()
        for a in ARTICLES:
            if p_low.startswith(a):
                p = p[len(a):].strip()
                break

        if not p:
            return ""
        tok = p.split(" ")[0].strip()
        tok = re.sub(r"[^\wäöüß\-]+", "", tok, flags=re.IGNORECASE)
        if len(tok) < 3 or tok.isdigit():
            return ""
        stop = {
            "der", "die", "das", "ein", "eine", "einen", "einem", "einer", "den", "dem", "des",
            "und", "oder", "zu", "im", "in", "am", "an", "von", "mit", "für", "fuer",
            "was", "wie", "warum", "wieso", "weshalb", "diese", "dieser", "dieses", "dabei",
        }
        if tok.lower() in stop:
            return ""
        if not tok[0].isalpha() or not tok[0].isupper():
            return ""
        return tok
