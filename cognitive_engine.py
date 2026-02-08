# -*- coding: utf-8 -*-
# ============================================================
# Cognitive Engine – Spreading Activation + Working Memory
# Datei: cognitive_engine.py
#
# VERSION: 3.1.4
# STATUS: STABIL (Pareto-Upgrade auf Gehirn-nahe Dynamik + Text-Import)
#
# ROLLE:
# - Kognitives Denksystem mit Spreading Activation (lokal über WM)
# - Entkopplung von Denken (internes Muster) und Sprechen (Output)
#
# VERANTWORTUNG:
# - Modell laden/speichern (Semantic + Episodic) als JSONL (UTF-8, Umlaute)
# - WM-basierte Ausbreitung (O(E_WM) statt O(N2))
# - Inhibition + Winner-Take-Most + Top-K Clamp
# - Intent-Gating (Was/Warum/Wie) für Relationstypen
# - Zeitabhängiger Decay (exp(-dt/tau))
# - Lernen: Hebbian + Anti-Hebbian light + Weight Decay (Episodic schnell vergessend)
# - Text-Import: einfache Extraktion (Regex) -> Nodes/Edges (für Learning-Interface)
#
# NICHT VERANTWORTLICH FÜR:
# - Große NLP-Pipelines, Embeddings, Transformer-Generierung
# - Fakten-Truth außerhalb der Trainingsdaten (kein Weltwissen „erfinden“)
#
# ÄNDERUNGEN:
# 3.1.4 – Text-Import-Lerninterface + Unknown-Stub + JSONL (UTF-8/Umlaute)
# 3.1.0  – WM + Inhibition + Intent-Gating + time-decay + Lernen + Episodic + Triangulation
# ============================================================

from __future__ import annotations

import json
import math
import os
import random
import re
import subprocess
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Tuple, Optional, Iterable


@dataclass
class Verbindung:
    ziel: str
    gewicht: float
    typ: str


@dataclass
class Konzept:
    id: str
    labels: List[str] = field(default_factory=list)
    semantische_features: List[str] = field(default_factory=list)
    verbindungen: List[Verbindung] = field(default_factory=list)  # semantic outbound
    aktivierung: float = 0.0
    letzte_aktivierung: datetime = field(default_factory=datetime.now)


@dataclass
class WMItem:
    id: str
    a: float
    role: str = "CONTEXT"   # FOCUS | CONTEXT
    age: int = 0


@dataclass
class TraceItem:
    tick: int
    src: str
    dst: str
    typ: str
    contrib: float
    layer: str  # "semantic" | "episodic"


class KognitivesModell:
    def __init__(
        self,
        modell_datei: str = "memory_semantic.jsonl",
        episodic_datei: Optional[str] = "memory_episodic.jsonl",
        lm_cmd: Optional[str] = None,
        lm_callable: Optional[Callable[[Dict[str, object]], str]] = None,
    ):
        self.konzepte: Dict[str, Konzept] = {}
        self.episodic_edges: Dict[str, List[Verbindung]] = {}
        self.trace: List[TraceItem] = []

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.semantic_datei = modell_datei
        self.episodic_datei = episodic_datei

        # Working Memory
        self.wm_max = 25
        self.wm_seed = 10
        self.wm: List[WMItem] = []

        # Spreading
        self.ticks = 7
        self.damp = 0.7

        # Zeitlicher Decay: exp(-dt/tau_seconds)
        self.tau_seconds = 6.0

        # Inhibition / Winner-Take-Most
        self.inhib_lambda = 0.15
        self.topk_global = 80
        self.cutoff = 0.01

        # Intent-Gating
        self.gate_match = 1.15
        self.gate_mismatch = 0.70

        # Pattern-Threshold (Antwort)
        self.pattern_threshold = 0.12

        # Lernen
        self.lr_hebb = 0.08
        self.lr_anti = 0.03
        self.weight_decay = 0.997

        # Episodic schneller vergessend
        self.episodic_weight_decay = 0.985

        # Triangulation
        self.triangle_boost = 1.05
        self.triangle_create_w = 0.18

        # Konsolidierung episodic -> semantic
        self.consolidate_threshold = 0.55
        self.consolidate_ratio = 0.7
        self.consolidate_boost = 0.04

        # Broca / LM (Hybrid)
        self.lm_cmd = lm_cmd or os.environ.get("LLM_CMD")
        self.lm_callable = lm_callable
        self.lm_timeout = 12
        self.use_lm_default = True
        self.explain_output = True
        self.broca_max_sents = 3

        if self.semantic_datei:
            self.modell_laden(self.semantic_datei, episodic=False)
        if self.episodic_datei:
            self.modell_laden(self.episodic_datei, episodic=True)

    # -------------------------
    # Unicode / Pfade / Normalisierung
    # -------------------------

    def _nfc(self, s: str) -> str:
        return unicodedata.normalize("NFC", s) if isinstance(s, str) else ""

    def _norm_label(self, s: str) -> str:
        t = self._nfc(s).strip().lower()
        t = re.sub(r"[\.,;:!?()\[\]{}<>\"'`]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        for a in ["der ", "die ", "das ", "ein ", "eine ", "einen ", "einem ", "einer ", "den ", "dem ", "des "]:
            if t.startswith(a):
                t = t[len(a):].strip()
                break
        return t

    def _ensure_label(self, kid: str, label: str):
        if kid not in self.konzepte:
            return
        lab = self._nfc(label).strip()
        if not lab:
            return
        labs = self.konzepte[kid].labels
        if lab not in labs and len(labs) < 10:
            labs.append(lab)

    def _label_for_output(self, kid: str) -> str:
        k = self.konzepte.get(kid)
        if not k or not k.labels:
            return kid
        labels = [l for l in k.labels if l]
        if not labels:
            return kid
        def score(lab: str) -> Tuple[int, int]:
            s = 0
            if " " in lab:
                s += 2
            if lab.lower() == lab:
                s += 1
            if lab != kid:
                s += 1
            return (s, len(lab))
        labels.sort(key=score, reverse=True)
        return labels[0]

    def _resolve_path(self, p: str) -> str:
        if not p:
            return p
        if os.path.isabs(p):
            return p
        return os.path.join(self.base_dir, p)

    def _canon_type(self, t: str) -> str:
        t0 = self._nfc(t).strip()
        if not t0:
            return ""
        mapping = {
            "entha": "enthält", "enthaelt": "enthält", "enthalt": "enthält",
            "beno": "benötigt", "benoetigt": "benötigt",
            "ermo": "ermöglicht", "ermoeglicht": "ermöglicht",
            "notwendig_fu": "notwendig_für", "notwendig_fuer": "notwendig_für",
            "durchla": "durchlässt", "durchlaesst": "durchlässt",
            "versta": "verstärkt", "verstaerkt": "verstärkt",
            "anfa": "anfällig_für", "anfaellig_fuer": "anfällig_für",
            "na": "nährt",
        }
        low = t0.lower()
        if low in mapping:
            return mapping[low]
        for k, v in mapping.items():
            if len(low) <= 12 and low.startswith(k):
                return v
        return t0

    # -------------------------
    # Intent / Gating
    # -------------------------

    def _erkenne_intent(self, frage: str) -> str:
        f = frage.strip().lower()
        if f.startswith(("warum", "weshalb", "wieso")):
            return "CAUSE"
        if f.startswith("wie"):
            return "HOW"
        if f.startswith("was"):
            return "DEF"
        return "OTHER"

    def _gate(self, edge_type: str, intent: str) -> float:
        t = edge_type.strip().lower()
        if intent == "DEF":
            preferred = {"ist", "teil_von", "klasse", "gehört_zu", "besteht_aus", "eigenschaft"}
        elif intent == "CAUSE":
            preferred = {"ermöglicht", "ermöglicht", "verursacht", "notwendig_für", "notwendig_für", "benötigt", "benötigt", "braucht"}
        elif intent == "HOW":
            preferred = {"prozess", "besteht_aus", "benötigt", "benötigt", "in", "von", "schritt", "lebt_in"}
        else:
            preferred = set()
        return self.gate_match if t in preferred else self.gate_mismatch

    # -------------------------
    # Modell laden (JSONL empfohlen)
    # -------------------------

    def modell_laden(self, datei: str, episodic: bool = False):
        if datei.lower().endswith(".jsonl"):
            return self._modell_laden_jsonl(datei, episodic=episodic)
        return self._modell_laden_txt(datei, episodic=episodic)

    def _modell_laden_txt(self, datei: str, episodic: bool = False):
        pfad = self._resolve_path(datei)
        try:
            with open(pfad, "r", encoding="utf-8") as f:
                for raw in f:
                    zeile = raw.strip()
                    if not zeile or zeile.startswith("#"):
                        continue
                    teile = [t.strip() for t in zeile.split("|")]
                    if len(teile) < 3:
                        continue
                    konzept_id = self._nfc(teile[0])
                    features = [self._nfc(x) for x in teile[1].split(",") if x.strip()]
                    verbindungs_str = teile[2].strip()

                    if konzept_id not in self.konzepte:
                        self.konzepte[konzept_id] = Konzept(id=konzept_id, labels=[konzept_id], semantische_features=features)
                    else:
                        if features:
                            existing = set(self.konzepte[konzept_id].semantische_features)
                            for ft in features:
                                if ft not in existing:
                                    self.konzepte[konzept_id].semantische_features.append(ft)
                                    existing.add(ft)
                        if not self.konzepte[konzept_id].labels:
                            self.konzepte[konzept_id].labels = [konzept_id]

                    edges = self._parse_verbindungen(verbindungs_str)
                    if episodic:
                        self.episodic_edges.setdefault(konzept_id, []).extend(edges)
                    else:
                        self.konzepte[konzept_id].verbindungen.extend(edges)

            layer = "episodic" if episodic else "semantic"
            print(f"✓ Modell geladen ({layer}): {datei} | Konzepte: {len(self.konzepte)}")
        except FileNotFoundError:
            layer = "episodic" if episodic else "semantic"
            print(f"❌ Modell-Datei nicht gefunden ({layer}): {datei}")

    def _modell_laden_jsonl(self, datei: str, episodic: bool = False):
        pfad = self._resolve_path(datei)
        try:
            with open(pfad, "r", encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        continue
                    t = obj.get("t")
                    if t == "meta":
                        continue
                    if t == "node":
                        cid = self._nfc(str(obj.get("id", "")))
                        if not cid:
                            continue
                        feats = obj.get("features") or []
                        feats = [self._nfc(str(x)) for x in feats if str(x).strip()]
                        labels = obj.get("labels") or []
                        labels = [self._nfc(str(x)) for x in labels if str(x).strip()]
                        if not labels:
                            labels = [cid]
                        if cid not in self.konzepte:
                            self.konzepte[cid] = Konzept(id=cid, labels=labels, semantische_features=feats)
                        else:
                            if feats:
                                existing = set(self.konzepte[cid].semantische_features)
                                for ft in feats:
                                    if ft not in existing:
                                        self.konzepte[cid].semantische_features.append(ft)
                                        existing.add(ft)
                            if labels:
                                existing_l = set(self.konzepte[cid].labels or [])
                                for lb in labels:
                                    if lb not in existing_l:
                                        self.konzepte[cid].labels.append(lb)
                                        existing_l.add(lb)
                        continue
                    if t == "edge":
                        src = self._nfc(str(obj.get("src", "")))
                        dst = self._nfc(str(obj.get("dst", "")))
                        if not src or not dst:
                            continue
                        try:
                            w = float(obj.get("w", 0.0))
                        except Exception:
                            w = 0.0
                        w = max(0.0, min(w, 0.99))
                        typ = self._canon_type(str(obj.get("type", "")))
                        if src not in self.konzepte:
                            self.konzepte[src] = Konzept(id=src, labels=[src])
                        if dst not in self.konzepte:
                            self.konzepte[dst] = Konzept(id=dst, labels=[dst])
                        edge = Verbindung(ziel=dst, gewicht=w, typ=typ)
                        if episodic:
                            self.episodic_edges.setdefault(src, []).append(edge)
                        else:
                            self.konzepte[src].verbindungen.append(edge)
                        continue

            layer = "episodic" if episodic else "semantic"
            print(f"✓ Modell geladen ({layer}): {datei} | Konzepte: {len(self.konzepte)}")
        except FileNotFoundError:
            layer = "episodic" if episodic else "semantic"
            print(f"❌ Modell-Datei nicht gefunden ({layer}): {datei}")

    def _parse_verbindungen(self, verbindungs_str: str) -> List[Verbindung]:
        out: List[Verbindung] = []
        if not verbindungs_str:
            return out
        for part in verbindungs_str.split(","):
            part = part.strip()
            if not part:
                continue
            m = re.match(r"^([^:]+):([0-9.]+):(.+)$", part)
            if not m:
                continue
            ziel = self._nfc(m.group(1).strip())
            try:
                gewicht = float(m.group(2))
            except Exception:
                continue
            typ = self._canon_type(m.group(3).strip())
            out.append(Verbindung(ziel=ziel, gewicht=max(0.0, min(gewicht, 0.99)), typ=typ))
        return out

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
    # Working Memory
    # -------------------------

    def _wm_init(self, cues: Dict[str, float]):
        self.wm = []
        items = sorted(cues.items(), key=lambda x: x[1], reverse=True)[: self.wm_seed]
        for i, (kid, a) in enumerate(items):
            role = "FOCUS" if i < 2 else "CONTEXT"
            self.wm.append(WMItem(id=kid, a=a, role=role, age=0))

    def _wm_refresh(self, act: Dict[str, float]):
        candidates = sorted(act.items(), key=lambda x: x[1], reverse=True)
        candidates = [(k, v) for k, v in candidates if v > self.cutoff]

        new_wm: List[WMItem] = []
        used = set()

        for k, v in candidates[:2]:
            new_wm.append(WMItem(id=k, a=v, role="FOCUS", age=0))
            used.add(k)

        for k, v in candidates[2:]:
            if len(new_wm) >= self.wm_max:
                break
            if k in used:
                continue
            new_wm.append(WMItem(id=k, a=v, role="CONTEXT", age=0))
            used.add(k)

        self.wm = new_wm

    # -------------------------
    # Zeit-Decay / Inhibition / Clamp
    # -------------------------

    def _time_decay(self, k: Konzept) -> float:
        dt = (datetime.now() - k.letzte_aktivierung).total_seconds()
        if dt <= 0:
            return 1.0
        return math.exp(-dt / max(0.001, self.tau_seconds))

    def _normalize_max(self, act: Dict[str, float]) -> Dict[str, float]:
        if not act:
            return act
        mx = max(act.values())
        if mx <= 0:
            return act
        if mx > 1.0:
            return {k: v / mx for k, v in act.items()}
        return act

    def _inhibit(self, act: Dict[str, float]) -> Dict[str, float]:
        if not act:
            return act
        mean_val = sum(act.values()) / max(1, len(act))
        lam = self.inhib_lambda
        return {k: max(0.0, v - lam * mean_val) for k, v in act.items()}

    def _topk_clamp(self, act: Dict[str, float], k: int) -> Dict[str, float]:
        if not act or k <= 0:
            return act
        items = sorted(act.items(), key=lambda x: x[1], reverse=True)[:k]
        return {a: b for a, b in items if b > 0.0}

    # -------------------------
    # Spreading Activation (lokal über WM)
    # -------------------------

    def _iter_edges(self, src: str) -> Iterable[Tuple[str, Verbindung, str]]:
        if src in self.konzepte:
            for e in self.konzepte[src].verbindungen:
                yield src, e, "semantic"
        if src in self.episodic_edges:
            for e in self.episodic_edges[src]:
                yield src, e, "episodic"

    def _has_edge(self, src: str, dst: str) -> bool:
        if src in self.konzepte:
            for e in self.konzepte[src].verbindungen:
                if e.ziel == dst:
                    return True
        if src in self.episodic_edges:
            for e in self.episodic_edges[src]:
                if e.ziel == dst:
                    return True
        return False

    def _triangle_adjust(self, src: str, dst: str) -> float:
        wm_ids = {w.id for w in self.wm}
        for mid in wm_ids:
            if mid == src or mid == dst:
                continue
            if self._has_edge(src, mid) and self._has_edge(mid, dst):
                return self.triangle_boost
        return 1.0

    def spreading_activation(self, intent: str = "OTHER") -> Dict[str, float]:
        act: Dict[str, float] = {}
        self.trace = []

        for w in self.wm:
            act[w.id] = max(act.get(w.id, 0.0), w.a)
            if w.id in self.konzepte:
                self.konzepte[w.id].aktivierung = act[w.id]
                self.konzepte[w.id].letzte_aktivierung = datetime.now()

        for tick in range(self.ticks):
            nxt: Dict[str, float] = dict(act)

            wm_ids = [w.id for w in self.wm]

            for src in wm_ids:
                src_a = act.get(src, 0.0)
                if src_a <= self.cutoff:
                    continue

                for _, e, layer in self._iter_edges(src):
                    dst = e.ziel
                    if not dst:
                        continue

                    gate = self._gate(e.typ, intent)
                    tri = self._triangle_adjust(src, dst)
                    incoming = src_a * e.gewicht * self.damp * gate * tri

                    if dst in self.konzepte:
                        incoming *= self._time_decay(self.konzepte[dst])

                    if incoming > 0:
                        nxt[dst] = nxt.get(dst, 0.0) + incoming

                    if incoming > 0.05:
                        self.trace.append(TraceItem(
                            tick=tick + 1,
                            src=src,
                            dst=dst,
                            typ=e.typ,
                            contrib=float(incoming),
                            layer=layer,
                        ))

            nxt = self._normalize_max(nxt)
            nxt = self._inhibit(nxt)
            nxt = self._topk_clamp(nxt, self.topk_global)

            now = datetime.now()
            for kid, val in nxt.items():
                if kid not in self.konzepte:
                    self.konzepte[kid] = Konzept(id=kid, labels=[kid])
                self.konzepte[kid].aktivierung = val
                self.konzepte[kid].letzte_aktivierung = now

            act = nxt
            self._wm_refresh(act)

        return act

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
    # Denken / Antworten
    # -------------------------

    def denken(self, frage: str) -> Dict:
        intent = self._erkenne_intent(frage)
        cues = self._cue_set(frage)

        if not cues:
            unknown = self._create_unknown_stub(frage)
            return {
                "frage": frage,
                "intent": intent,
                "unknown": unknown,
                "denkmuster": [],
                "trace": [],
                "timestamp": datetime.now().isoformat(),
            }

        self._wm_init(cues)
        act = self.spreading_activation(intent=intent)

        pattern = sorted(
            [(k, v) for k, v in act.items() if v >= self.pattern_threshold],
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            "frage": frage,
            "intent": intent,
            "unknown": "",
            "denkmuster": pattern,
            "trace": sorted(self.trace, key=lambda t: t.contrib, reverse=True)[:20],
            "timestamp": datetime.now().isoformat(),
        }

    def antworte(self, frage: str, auto_lernen: bool = True, use_lm: Optional[bool] = None) -> str:
        res = self.denken(frage)
        pattern = res["denkmuster"]
        trace = res.get("trace") or []

        if not pattern:
            unk = res.get("unknown") or ""
            if unk:
                return (
                    f"Ich kenne {unk} noch nicht. "
                    "Gib mir einen kurzen Text (oder eine Definition) zum Import, dann kann ich es lernen. "
                    "Tipp: /import <pfad_zur_txt_datei> (oder CLI: --import <pfad>)."
                )
            return "Ich weiß das nicht."

        if auto_lernen:
            self.lerne_aus_aktivierung(pattern, trace)

        use_lm_final = self.use_lm_default if use_lm is None else use_lm
        return self.versprachliche(pattern, intent=res["intent"], trace=trace, use_lm=use_lm_final)

    # -------------------------
    # Broca: Muster -> Satz
    # -------------------------

    def versprachliche(
        self,
        denkmuster: List[Tuple[str, float]],
        intent: str = "OTHER",
        trace: Optional[List[TraceItem]] = None,
        use_lm: bool = True,
    ) -> str:
        if not denkmuster:
            return "Ich weiß das nicht."
        focus = [k for k, _ in denkmuster[:2]]
        context = [k for k, _ in denkmuster[2:8]]

        if use_lm:
            lm_text = self._lm_generate(denkmuster, intent=intent, trace=trace)
            if lm_text:
                if self.explain_output and trace:
                    t0 = trace[0]
                    lm_text = lm_text.rstrip() + f" (Trace: {t0.src} → {t0.dst} / {t0.typ})"
                return lm_text

        sentences: List[str] = []
        for rel in self._rank_candidate_edges(denkmuster, intent=intent)[: self.broca_max_sents]:
            tpl = self._template_for_type(rel["typ"])
            sentences.append(tpl.format(self._label_for_output(rel["src"]), self._label_for_output(rel["dst"])))

        if not sentences and denkmuster:
            sentences.append(f"Das zentrale Konzept ist {self._label_for_output(denkmuster[0][0])}.")

        extra = [self._label_for_output(k) for k in context if k not in focus][:3]
        if extra:
            sentences.append("Daneben sind auch " + ", ".join(extra) + " relevant.")

        s1 = " ".join(sentences)
        if self.explain_output and trace:
            t0 = trace[0]
            s1 = s1.rstrip() + f" (Trace: {t0.src} → {t0.dst} / {t0.typ})"
        return s1

    def _template_for_type(self, typ: str) -> str:
        templates = {
            "teil_von": "{} ist ein wesentlicher Teil von {}.",
            "hat": "{} hat oder besitzt {}.",
            "ist": "{} ist im Grunde {}.",
            "eigenschaft": "{} hat die charakteristische Eigenschaft {}.",
            "prozess": "{} ist ein Prozess, in dem {} zentral ist.",
            "ermöglicht": "{} ermöglicht {}.",
            "besteht_aus": "{} setzt sich zusammen aus {}.",
            "benötigt": "{} benötigt {} als Voraussetzung.",
            "braucht": "{} braucht {} zum Funktionieren.",
            "verursacht": "{} verursacht {}.",
            "notwendig_für": "{} ist notwendig für {}.",
            "gehört_zu": "{} gehört zu {}.",
            "lebt_in": "{} lebt in {}.",
            "gelernt": "{} und {} stehen in enger Beziehung.",
            "assoziation": "{} steht in Zusammenhang mit {}.",
        }
        return templates.get(typ, "{} und {} sind eng miteinander verbunden.")

    def _rank_candidate_edges(self, denkmuster: List[Tuple[str, float]], intent: str) -> List[Dict[str, object]]:
        aktive = {k for k, _ in denkmuster}
        out: List[Dict[str, object]] = []
        for src in aktive:
            for e in self.konzepte.get(src, Konzept(src)).verbindungen:
                if e.ziel in aktive:
                    score = e.gewicht * self._gate(e.typ, intent)
                    out.append({"score": score, "src": src, "dst": e.ziel, "typ": e.typ, "layer": "semantic"})
            for e in self.episodic_edges.get(src, []):
                if e.ziel in aktive:
                    score = (e.gewicht * 0.9) * self._gate(e.typ, intent)
                    out.append({"score": score, "src": src, "dst": e.ziel, "typ": e.typ, "layer": "episodic"})
        out.sort(key=lambda x: x["score"], reverse=True)
        return out

    def _lm_generate(self, denkmuster: List[Tuple[str, float]], intent: str, trace: Optional[List[TraceItem]]) -> Optional[str]:
        if not (self.lm_callable or self.lm_cmd):
            return None
        aktive = [self._label_for_output(k) for k, _ in denkmuster[:8]]
        rels = self._rank_candidate_edges(denkmuster, intent=intent)[:5]
        rel_lines = [
            f"{self._label_for_output(r['src'])} -{r['typ']}-> {self._label_for_output(r['dst'])} (w={r['score']:.2f}, {r['layer']})"
            for r in rels
        ]
        prompt = (
            "Du bist Broca und formulierst kurze, natürliche deutsche Sätze.\n"
            "Nutze die folgenden Konzepte und Relationen, erfinde keine neuen Fakten.\n"
            "Gib 1 bis 3 Sätze aus, keine Listen.\n\n"
            f"Intent: {intent}\n"
            f"Konzepte: {', '.join(aktive)}\n"
            "Relationen:\n" + "\n".join(rel_lines) + "\n"
        )

        if self.lm_callable:
            try:
                txt = self.lm_callable({"prompt": prompt, "intent": intent, "concepts": aktive, "relations": rels, "trace": trace})
                return txt.strip() if isinstance(txt, str) and txt.strip() else None
            except Exception:
                return None

        try:
            result = subprocess.run(
                self.lm_cmd,
                input=prompt,
                text=True,
                capture_output=True,
                shell=True,
                timeout=self.lm_timeout,
            )
            out = (result.stdout or "").strip()
            if out:
                return out
        except Exception:
            return None
        return None

    # -------------------------
    # Lernen (Hebb + Anti-Hebb + Decay)
    # -------------------------

    def _get_or_create_episodic_edge(self, src: str, dst: str, typ: str, w_init: float) -> Verbindung:
        self.episodic_edges.setdefault(src, [])
        for e in self.episodic_edges[src]:
            if e.ziel == dst and e.typ == typ:
                return e
        e = Verbindung(ziel=dst, gewicht=max(0.01, min(w_init, 0.99)), typ=typ)
        self.episodic_edges[src].append(e)
        return e

    def lerne_aus_aktivierung(self, pattern: List[Tuple[str, float]], trace: List[TraceItem]):
        aktive = {k for k, _ in pattern}
        act_map = {k: a for k, a in pattern}

        for t in trace[:12]:
            if t.contrib <= 0:
                continue
            w0 = 0.25 + min(0.6, t.contrib)
            e = self._get_or_create_episodic_edge(t.src, t.dst, self._canon_type(t.typ), w_init=w0)
            e.gewicht = min(0.99, e.gewicht + self.lr_hebb * max(act_map.get(t.src, 0.0), act_map.get(t.dst, 0.0)))

        act_sorted = sorted(pattern, key=lambda x: x[1], reverse=True)[:10]
        ids = [k for k, _ in act_sorted]
        for a in ids:
            for b in ids:
                if a == b:
                    continue
                for c in ids:
                    if c in (a, b):
                        continue
                    if self._has_edge(a, b) and self._has_edge(b, c) and not self._has_edge(a, c):
                        self._get_or_create_episodic_edge(a, c, "assoziation", w_init=self.triangle_create_w)

        for src, src_a in act_map.items():
            if src_a < self.pattern_threshold:
                continue
            k = self.konzepte.get(src)
            if not k:
                continue
            for e in k.verbindungen:
                if e.ziel not in aktive:
                    e.gewicht = max(0.01, e.gewicht - self.lr_anti * src_a)

        for k in self.konzepte.values():
            for e in k.verbindungen:
                e.gewicht = max(0.01, min(0.99, e.gewicht * self.weight_decay))

        for src, edges in list(self.episodic_edges.items()):
            kept = []
            for e in edges:
                e.gewicht = max(0.01, min(0.99, e.gewicht * self.episodic_weight_decay))
                if e.gewicht >= 0.06:
                    kept.append(e)
            self.episodic_edges[src] = kept

        self.konsolidiere_episodisch()

    def _get_or_create_semantic_edge(self, src: str, dst: str, typ: str, w_init: float) -> Verbindung:
        if src not in self.konzepte:
            self.konzepte[src] = Konzept(id=src, labels=[src])
        for e in self.konzepte[src].verbindungen:
            if e.ziel == dst and e.typ == typ:
                return e
        e = Verbindung(ziel=dst, gewicht=max(0.01, min(w_init, 0.99)), typ=typ)
        self.konzepte[src].verbindungen.append(e)
        return e

    def konsolidiere_episodisch(self):
        for src, edges in self.episodic_edges.items():
            for e in edges:
                if e.gewicht < self.consolidate_threshold:
                    continue
                se = self._get_or_create_semantic_edge(src, e.ziel, self._canon_type(e.typ), w_init=e.gewicht * self.consolidate_ratio)
                se.gewicht = min(0.99, max(se.gewicht, e.gewicht * self.consolidate_ratio))
                se.gewicht = min(0.99, se.gewicht + self.consolidate_boost)

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
                self.konzepte["Import"] = Konzept(id="Import", semantische_features=["Meta"])
                nodes_added += 1
            if tag not in self.konzepte["Import"].semantische_features:
                self.konzepte["Import"].semantische_features.append(tag)

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

    # -------------------------
    # Speichern (JSONL empfohlen)
    # -------------------------

    def speichere_model(self, datei: str, episodic_datei: Optional[str] = None):
        if datei.lower().endswith(".jsonl"):
            self._speichere_jsonl(datei, episodic_datei=episodic_datei)
        else:
            self._speichere_txt(datei, episodic_datei=episodic_datei)

    def _speichere_jsonl(self, datei: str, episodic_datei: Optional[str] = None):
        base = self._resolve_path(datei)
        meta = {
            "t": "meta",
            "format": "memory_jsonl",
            "encoding": "utf-8",
            "umlaute": True,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "counts": {
                "nodes": len(self.konzepte),
                "edges": sum(len(k.verbindungen) for k in self.konzepte.values()),
            },
        }
        with open(base, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            for cid in sorted(self.konzepte.keys()):
                feats = list(self.konzepte[cid].semantische_features or [])
                labels = list(self.konzepte[cid].labels or [])
                if not labels:
                    labels = [cid]
                obj = {"t": "node", "id": cid, "features": sorted(set(feats)), "labels": sorted(set(labels))}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            for src in sorted(self.konzepte.keys()):
                for v in self.konzepte[src].verbindungen:
                    obj = {
                        "t": "edge",
                        "src": src,
                        "dst": v.ziel,
                        "w": round(float(v.gewicht), 4),
                        "type": self._canon_type(v.typ),
                    }
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        if episodic_datei:
            epi_path = self._resolve_path(episodic_datei)
            meta2 = {
                "t": "meta",
                "format": "memory_jsonl",
                "encoding": "utf-8",
                "umlaute": True,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "counts": {
                    "nodes": len(self.konzepte),
                    "edges": sum(len(e) for e in self.episodic_edges.values()),
                },
            }
            with open(epi_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(meta2, ensure_ascii=False) + "\n")
                for cid in sorted(self.konzepte.keys()):
                    feats = list(self.konzepte[cid].semantische_features or [])
                    labels = list(self.konzepte[cid].labels or [])
                    if not labels:
                        labels = [cid]
                    obj = {"t": "node", "id": cid, "features": sorted(set(feats)), "labels": sorted(set(labels))}
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                for src in sorted(self.episodic_edges.keys()):
                    for v in self.episodic_edges[src]:
                        obj = {
                            "t": "edge",
                            "src": src,
                            "dst": v.ziel,
                            "w": round(float(v.gewicht), 4),
                            "type": self._canon_type(v.typ),
                        }
                        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        print(f"✓ Modelle gespeichert: {datei}" + (f" + {episodic_datei}" if episodic_datei else ""))

    def _speichere_txt(self, datei: str, episodic_datei: Optional[str] = None):
        with open(self._resolve_path(datei), "w", encoding="utf-8") as f:
            f.write("# KOGNITIVES MODELL (SEMANTIC) - AUTO-GENERIERT\n")
            f.write("# Format: konzept | features | verbindungen\n\n")
            for konzept_id, konzept in sorted(self.konzepte.items()):
                features = ",".join(konzept.semantische_features) if konzept.semantische_features else "gelernt"
                verbindungen = ", ".join([f"{v.ziel}:{v.gewicht:.2f}:{self._canon_type(v.typ)}" for v in konzept.verbindungen])
                f.write(f"{konzept_id} | {features} | {verbindungen}\n")

        if episodic_datei:
            with open(self._resolve_path(episodic_datei), "w", encoding="utf-8") as f:
                f.write("# KOGNITIVES MODELL (EPISODIC) - AUTO-GENERIERT\n")
                f.write("# Format: konzept | features | verbindungen\n\n")
                for src, edges in sorted(self.episodic_edges.items()):
                    verbindungen = ", ".join([f"{v.ziel}:{v.gewicht:.2f}:{self._canon_type(v.typ)}" for v in edges])
                    f.write(f"{src} | episodic | {verbindungen}\n")

        print(f"✓ Modelle gespeichert: {datei}" + (f" + {episodic_datei}" if episodic_datei else ""))

    # -------------------------
    # Autonomes Denken
    # -------------------------

    def autonom_denken(self, steps: int = 1) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        steps = max(1, min(steps, 10))

        for _ in range(steps):
            candidates = []
            for kid, k in self.konzepte.items():
                sem_strength = sum(e.gewicht for e in k.verbindungen)
                epi_strength = sum(e.gewicht for e in self.episodic_edges.get(kid, []))
                novelty = 1.0 / (1.0 + sem_strength + epi_strength)
                gap = max(0.0, epi_strength - sem_strength)
                score = gap + 0.6 * novelty + random.uniform(0.0, 0.05)
                candidates.append((kid, score))

            candidates.sort(key=lambda x: x[1], reverse=True)
            seeds = [kid for kid, _ in candidates[:2]]
            if not seeds:
                break

            cues = {seeds[0]: 0.95}
            if len(seeds) > 1:
                cues[seeds[1]] = 0.75

            self._wm_init(cues)
            act = self.spreading_activation(intent="OTHER")
            pattern = sorted(
                [(k, v) for k, v in act.items() if v >= self.pattern_threshold],
                key=lambda x: x[1],
                reverse=True,
            )

            res = {
                "mode": "autonomous",
                "seeds": seeds,
                "denkmuster": pattern,
                "trace": sorted(self.trace, key=lambda t: t.contrib, reverse=True)[:20],
                "timestamp": datetime.now().isoformat(),
            }
            if pattern:
                self.lerne_aus_aktivierung(pattern, res["trace"])
            results.append(res)

        return results


# ============================================================
# ROWCOUNT FOOTER
# ============================================================
# ROWCOUNT_OLD: 664
# ROWCOUNT_NEW: 987
