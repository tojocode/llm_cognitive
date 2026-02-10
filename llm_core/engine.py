# -*- coding: utf-8 -*-
# ============================================================
# Cognitive Engine – Spreading Activation + Working Memory
# Datei: llm_core/engine.py
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
import unicodedata
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from llm_core.types import Konzept, TraceItem, Verbindung, WMItem
from llm_memory.embedding_db import EmbeddingDB
from llm_speech import BrocaMixin, WernickeMixin


class KognitivesModell(WernickeMixin, BrocaMixin):
    def __init__(
        self,
        modell_datei: str = "llm_memory/memory_semantic.jsonl",
        episodic_datei: Optional[str] = "llm_memory/memory_episodic.jsonl",
        lm_cmd: Optional[str] = None,
        lm_callable: Optional[Callable[[Dict[str, object]], str]] = None,
        lexikon_datei: Optional[str] = "llm_memory/lexikon.json",
        embed_db_path: Optional[str] = "llm_memory/embeddings.json",
        embed_dim: int = 512,
    ):
        self.konzepte: Dict[str, Konzept] = {}
        self.episodic_edges: Dict[str, List[Verbindung]] = {}
        self.trace: List[TraceItem] = []

        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.semantic_datei = modell_datei
        self.episodic_datei = episodic_datei

        # Working Memory
        self.wm_max = 25
        self.wm_seed = 10
        self.wm: List[WMItem] = []

        # Spreading
        self.ticks = 7
        self.damp = 0.7

        # Predictive Activation
        self.pred_enabled = True
        self.pred_lr = 0.08
        self.pred_error_gain = 0.35
        self.pred_min_error = 0.02
        self.pred_learning_only = True

        # Sequenz-Kanten (zeitlicher Kontext)
        self.seq_type = "folge"
        self.seq_lr = 0.12
        self.seq_boost = 1.1
        self.seq_min_act = 0.2

        # Global Workspace Gate
        self.workspace_gate = True
        self.workspace_topk = 6
        self.workspace_focus_boost = 1.8

        # Zeitlicher Decay: exp(-dt/tau_seconds)
        self.tau_seconds = 6.0

        # Inhibition / Winner-Take-Most
        self.inhib_lambda = 0.22
        self.inhib_degree_boost = 0.8
        self.inhib_degree_norm = 10.0
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

        # Sprache (Broca/Wernicke)
        self._init_broca(lm_cmd, lm_callable)
        self._init_wernicke(lexikon_datei)

        # Hybrid Memory (Embedding DB)
        self.embed_enabled = True
        self.embed_top_k = 3
        self.embed_min_score = 0.18
        self.embed_min_score_strong = 0.3
        self.embed_cue_boost = 0.45
        self.embed_store_on_import = True
        self.embed_db_path = embed_db_path
        self.embed_db = None

        # Goal Layer
        self.goal_state: List[str] = []
        self.goal_boost = 0.55
        self.goal_neighbor_boost = 0.35

        # Autonomes Denken (Diversität)
        self.think_topk = 40
        self.think_recent_max = 24
        self.think_cooldown = 8
        self.think_link_penalty = 0.6
        self.think_low_degree_penalty = 0.25
        self._think_recent: List[str] = []

        if self.semantic_datei:
            self.modell_laden(self.semantic_datei, episodic=False)
        if self.episodic_datei:
            self.modell_laden(self.episodic_datei, episodic=True)
        # Lexikon wird in _init_wernicke geladen
        self._init_embeddings(embed_db_path, embed_dim)

    # -------------------------
    # Unicode / Pfade / Normalisierung
    # -------------------------

    def _nfc(self, s: str) -> str:
        return unicodedata.normalize("NFC", s) if isinstance(s, str) else ""

    def _ensure_label(self, kid: str, label: str):
        if kid not in self.konzepte:
            return
        lab = self._nfc(label).strip()
        if not lab:
            return
        labs = self.konzepte[kid].labels
        if lab not in labs and len(labs) < 10:
            labs.append(lab)

    def _resolve_path(self, p: str) -> str:
        if not p:
            return p
        if os.path.isabs(p):
            return p
        return os.path.join(self.base_dir, p)

    # -------------------------
    # Hybrid Memory (Embedding DB)
    # -------------------------

    def _init_embeddings(self, path: Optional[str], dim: int):
        if not path:
            self.embed_db = None
            return
        try:
            ep = self._resolve_path(path)
            self.embed_db = EmbeddingDB(ep, dim=dim)
        except Exception:
            self.embed_db = None

    def embed_add_text(self, text: str, meta: Optional[Dict[str, object]] = None) -> bool:
        if not self.embed_enabled or not self.embed_db:
            return False
        return self.embed_db.add(text, meta=meta)

    def embed_query(self, text: str) -> List[Dict[str, object]]:
        if not self.embed_enabled or not self.embed_db:
            return []
        return self.embed_db.query(text, top_k=self.embed_top_k, min_score=self.embed_min_score)

    def speichere_embeddings(self):
        if self.embed_db:
            self.embed_db.save()

    # -------------------------
    # Goal Layer
    # -------------------------

    def set_goals(self, goals: List[str]):
        self.goal_state = [g.strip() for g in (goals or []) if g and str(g).strip()]

    def add_goal(self, goal: str):
        g = (goal or "").strip()
        if not g:
            return
        if g not in self.goal_state:
            self.goal_state.append(g)

    def clear_goals(self):
        self.goal_state = []

    def _goal_ids(self) -> List[str]:
        ids: List[str] = []
        for g in self.goal_state:
            try:
                cid = self._phrase_to_concept_id(g)
            except Exception:
                cid = ""
            if cid and cid in self.konzepte and cid not in ids:
                ids.append(cid)
        return ids

    def _goal_cues(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for gid in self._goal_ids():
            out[gid] = max(out.get(gid, 0.0), self.goal_boost)
            for e in self.konzepte.get(gid, Konzept(gid)).verbindungen:
                out[e.ziel] = max(out.get(e.ziel, 0.0), self.goal_neighbor_boost * e.gewicht)
            for e in self.episodic_edges.get(gid, []):
                out[e.ziel] = max(out.get(e.ziel, 0.0), self.goal_neighbor_boost * 0.9 * e.gewicht)
        return out


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
        if f.startswith(("warum", "weshalb", "wieso", "wodurch", "womit")):
            return "CAUSE"
        if f.startswith(("wo ", "wohin", "woher")):
            return "WHERE"
        if f.startswith(("woraus", "woraus besteht", "woraus setzt", "woraus besteht")):
            return "PARTS"
        if f.startswith(("welche eigenschaften", "welche merkmale", "welche eigenschaft")):
            return "PROPS"
        if f.startswith("wie"):
            return "HOW"
        if f.startswith("was"):
            return "DEF"
        if f.startswith(("ist ", "sind ", "hat ", "haben ")):
            return "DEF"
        return "OTHER"

    def _gate(self, edge_type: str, intent: str) -> float:
        t = edge_type.strip().lower()
        if intent == "DEF":
            preferred = {
                "ist",
                "teil_von",
                "klasse",
                "gehört_zu",
                "besteht_aus",
                "eigenschaft",
            }
        elif intent == "CAUSE":
            preferred = {
                "ermöglicht",
                "verursacht",
                "verursacht_durch",
                "verursacht_von",
                "notwendig_für",
                "benötigt",
                "braucht",
            }
        elif intent == "WHERE":
            preferred = {"lebt_in", "in", "von"}
        elif intent == "PARTS":
            preferred = {"besteht_aus", "enthält", "hat", "teil_von"}
        elif intent == "PROPS":
            preferred = {"eigenschaft", "eigenschaft_von", "hat", "ist"}
        elif intent == "HOW":
            preferred = {
                "prozess",
                "besteht_aus",
                "benötigt",
                "in",
                "von",
                "schritt",
                "lebt_in",
            }
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
                        self.konzepte[konzept_id] = Konzept(
                            id=konzept_id,
                            labels=[konzept_id],
                            semantische_features=features,
                        )
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
            print(f"OK Modell geladen ({layer}): {datei} | Konzepte: {len(self.konzepte)}")
        except FileNotFoundError:
            layer = "episodic" if episodic else "semantic"
            print(f"FEHLER Modell-Datei nicht gefunden ({layer}): {datei}")

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
                            self.konzepte[cid] = Konzept(
                                id=cid,
                                labels=labels,
                                semantische_features=feats,
                            )
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
            print(f"OK Modell geladen ({layer}): {datei} | Konzepte: {len(self.konzepte)}")
        except FileNotFoundError:
            layer = "episodic" if episodic else "semantic"
            print(f"FEHLER Modell-Datei nicht gefunden ({layer}): {datei}")

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
    # Working Memory
    # -------------------------

    def _wm_init(self, cues: Dict[str, float]):
        self.wm = []
        items = sorted(cues.items(), key=lambda x: x[1], reverse=True)[: self.wm_seed]
        for i, (kid, a) in enumerate(items):
            role = "FOCUS" if i < 2 else "CONTEXT"
            self.wm.append(WMItem(id=kid, a=a, role=role))

    def _wm_refresh(self, act: Dict[str, float]):
        candidates = sorted(act.items(), key=lambda x: x[1], reverse=True)
        candidates = [(k, v) for k, v in candidates if v > self.cutoff]

        new_wm: List[WMItem] = []
        used = set()

        for k, v in candidates[:2]:
            new_wm.append(WMItem(id=k, a=v, role="FOCUS"))
            used.add(k)

        for k, v in candidates[2:]:
            if len(new_wm) >= self.wm_max:
                break
            if k in used:
                continue
            new_wm.append(WMItem(id=k, a=v, role="CONTEXT"))
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
        out = {}
        for k, v in act.items():
            deg_scale = 1.0
            if self.inhib_degree_boost > 0:
                deg = self._degree(k)
                norm = max(1.0, float(self.inhib_degree_norm))
                deg_scale = 1.0 + min(1.0, deg / norm) * self.inhib_degree_boost
            out[k] = max(0.0, v - lam * mean_val * deg_scale)
        return out

    def _degree(self, kid: str) -> int:
        deg = 0
        k = self.konzepte.get(kid)
        if k:
            deg += len(k.verbindungen)
        deg += len(self.episodic_edges.get(kid, []))
        return deg

    def _topk_clamp(self, act: Dict[str, float], k: int) -> Dict[str, float]:
        if not act or k <= 0:
            return act
        items = sorted(act.items(), key=lambda x: x[1], reverse=True)[:k]
        return {a: b for a, b in items if b > 0.0}

    # -------------------------
    # Spreading Activation (lokal über WM)
    # -------------------------

    def _iter_edges(
        self,
        src: str,
        include_seq: bool = True,
    ) -> Iterable[Tuple[str, Verbindung, str]]:
        if src in self.konzepte:
            for e in self.konzepte[src].verbindungen:
                yield src, e, "semantic"
        if src in self.episodic_edges:
            for e in self.episodic_edges[src]:
                if not include_seq and e.typ == self.seq_type:
                    continue
                yield src, e, "episodic"

    def _has_edge(self, src: str, dst: str) -> bool:
        if src in self.konzepte:
            for e in self.konzepte[src].verbindungen:
                if e.ziel == dst:
                    return True
        if src in self.episodic_edges:
            for e in self.episodic_edges[src]:
                if e.typ == self.seq_type:
                    continue
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

    def _update_seq_edge(self, src: str, dst: str, strength: float):
        if not src or not dst or src == dst:
            return
        w_init = max(0.1, min(0.9, float(strength or 0.0)))
        e = self._get_or_create_episodic_edge(src, dst, self.seq_type, w_init=w_init)
        e.gewicht = max(0.01, min(0.99, e.gewicht + self.seq_lr * max(0.05, w_init)))

    def spreading_activation(self, intent: str = "OTHER") -> Dict[str, float]:
        act: Dict[str, float] = {}
        self.trace = []

        for w in self.wm:
            act[w.id] = max(act.get(w.id, 0.0), w.a)
            if w.id in self.konzepte:
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
                self.konzepte[kid].letzte_aktivierung = now

            act = nxt
            self._wm_refresh(act)

        return act

    def predictive_activation(
        self,
        cues: Dict[str, float],
        intent: str = "OTHER",
    ) -> Dict[str, float]:
        act: Dict[str, float] = {}
        self.trace = []

        self._wm_init(cues)
        for w in self.wm:
            act[w.id] = max(act.get(w.id, 0.0), w.a)
            if w.id in self.konzepte:
                self.konzepte[w.id].letzte_aktivierung = datetime.now()

        prev_focus = self.wm[0].id if self.wm else ""

        for tick in range(self.ticks):
            pred: Dict[str, float] = {}
            wm_ids = [w.id for w in self.wm]

            for src in wm_ids:
                src_a = act.get(src, 0.0)
                if src_a <= self.cutoff:
                    continue

                for _, e, layer in self._iter_edges(src, include_seq=True):
                    dst = e.ziel
                    if not dst:
                        continue

                    gate = self._gate(e.typ, intent)
                    if layer == "episodic" and e.typ == self.seq_type:
                        gate = max(gate, self.seq_boost)

                    tri = self._triangle_adjust(src, dst)
                    incoming = src_a * e.gewicht * self.damp * gate * tri

                    if dst in self.konzepte:
                        incoming *= self._time_decay(self.konzepte[dst])

                    if incoming > 0:
                        pred[dst] = pred.get(dst, 0.0) + incoming

                    if incoming > 0.05:
                        self.trace.append(TraceItem(
                            tick=tick + 1,
                            src=src,
                            dst=dst,
                            typ=e.typ,
                            contrib=float(incoming),
                            layer=layer,
                        ))

            pred = self._normalize_max(pred)
            pred = self._inhibit(pred)
            pred = self._topk_clamp(pred, self.topk_global)

            # Prediction error (observed - predicted)
            error: Dict[str, float] = {}
            for k in set(pred) | set(act):
                error[k] = act.get(k, 0.0) - pred.get(k, 0.0)

            # Lernupdate nur über Vorhersagefehler (ohne Sequenzkanten)
            for src in wm_ids:
                src_a = act.get(src, 0.0)
                if src_a <= self.cutoff:
                    continue
                for _, e, layer in self._iter_edges(src, include_seq=True):
                    if e.typ == self.seq_type:
                        continue
                    dst = e.ziel
                    if not dst:
                        continue
                    err = error.get(dst, 0.0)
                    if abs(err) < self.pred_min_error:
                        continue
                    delta = self.pred_lr * src_a * err
                    e.gewicht = max(0.01, min(0.99, e.gewicht + delta))

            # Next activation = prediction + sensory anchors + (optional) error injection
            act = dict(pred)
            for k, v in cues.items():
                act[k] = max(act.get(k, 0.0), v)
            if self.pred_error_gain > 0:
                for k, err in error.items():
                    if err > 0:
                        val = min(0.99, err * self.pred_error_gain)
                        if val > act.get(k, 0.0):
                            act[k] = val

            now = datetime.now()
            for kid, val in act.items():
                if kid not in self.konzepte:
                    self.konzepte[kid] = Konzept(id=kid, labels=[kid])
                self.konzepte[kid].letzte_aktivierung = now

            self._wm_refresh(act)

            # Sequenzkanten updaten (Focus -> Focus)
            cur_focus = self.wm[0].id if self.wm else ""
            if prev_focus and cur_focus and prev_focus != cur_focus:
                strength = act.get(cur_focus, 0.0)
                if strength >= self.seq_min_act:
                    self._update_seq_edge(prev_focus, cur_focus, strength)
            prev_focus = cur_focus

        return act

    # -------------------------
    # Denken / Antworten
    # -------------------------

    def denken(self, frage: str) -> Dict:
        intent = self._erkenne_intent(frage)
        if self._is_greeting(frage):
            return {
                "frage": frage,
                "intent": intent,
                "unknown": "",
                "denkmuster": [],
                "focus": [],
                "memories": [],
                "trace": [],
                "smalltalk": "greeting",
                "timestamp": datetime.now().isoformat(),
            }

        base_cues = self._cue_set(frage)
        cues = dict(base_cues)
        seed_set = set(base_cues.keys())
        focus_ids = [k for k, _ in sorted(cues.items(), key=lambda x: x[1], reverse=True)[:2]]
        memory_hits: List[Dict[str, object]] = []

        # Hybrid memory: retrieve similar texts and use as weak cues
        if self.embed_enabled and self._use_embeddings_for(frage):
            memory_hits = self.embed_query(frage)
            top_score = float(memory_hits[0].get("score") or 0.0) if memory_hits else 0.0
            if not base_cues and top_score < self.embed_min_score_strong:
                memory_hits = []
            for hit in memory_hits:
                score = float(hit.get("score") or 0.0)
                if score <= 0:
                    continue
                mcues = self._cue_set(str(hit.get("text") or ""))
                boost = self.embed_cue_boost * score
                for k, v in mcues.items():
                    cues[k] = max(cues.get(k, 0.0), min(0.99, v * boost))

        # Goal Layer cues (soft bias)
        for k, v in self._goal_cues().items():
            cues[k] = max(cues.get(k, 0.0), v)

        if not cues:
            unknown = self._create_unknown_stub(frage)
            return {
                "frage": frage,
                "intent": intent,
                "unknown": unknown,
                "denkmuster": [],
                "focus": focus_ids,
                "memories": memory_hits,
                "trace": [],
                "timestamp": datetime.now().isoformat(),
            }

        if self.pred_enabled:
            act = self.predictive_activation(cues, intent=intent)
        else:
            self._wm_init(cues)
            act = self.spreading_activation(intent=intent)

        pattern_all = sorted(
            [(k, v) for k, v in act.items() if v >= self.pattern_threshold],
            key=lambda x: x[1],
            reverse=True,
        )

        if self.workspace_gate:
            wm_ids = [w.id for w in self.wm]
            wm_set = set(wm_ids)
            pattern = [(k, v) for k, v in pattern_all if k in wm_set]
            if not pattern and wm_ids:
                pattern = [
                    (k, act.get(k, 0.0))
                    for k in wm_ids
                    if act.get(k, 0.0) >= self.pattern_threshold
                ]
            def _ws_score(item: Tuple[str, float]) -> Tuple[int, float]:
                kid, val = item
                boosted = val * (self.workspace_focus_boost if kid in seed_set else 1.0)
                if intent == "DEF" and kid in seed_set:
                    return (1, boosted)
                return (0, boosted)

            pattern.sort(key=_ws_score, reverse=True)
            if self.workspace_topk and len(pattern) > self.workspace_topk:
                pattern = pattern[: self.workspace_topk]
        else:
            pattern = pattern_all

        # Focus should stay anchored to the question (seed_set), not drift in WM.
        focus_ids = self._focus_from_question(frage, seed_set)
        if not focus_ids and self.wm:
            focus_ids = [w.id for w in self.wm[:2]]
        elif self.wm and len(focus_ids) < 2:
            for w in self.wm:
                if w.id in focus_ids:
                    continue
                focus_ids.append(w.id)
                if len(focus_ids) >= 2:
                    break

        return {
            "frage": frage,
            "intent": intent,
            "unknown": "",
            "denkmuster": pattern,
            "focus": focus_ids,
            "memories": memory_hits,
            "trace": sorted(self.trace, key=lambda t: t.contrib, reverse=True)[:20],
            "timestamp": datetime.now().isoformat(),
        }

    def antworte(self, frage: str, auto_lernen: bool = True, use_lm: Optional[bool] = None) -> str:
        if self._is_greeting(frage):
            return "Das weiß ich nicht. Bitte stelle mir eine neue Frage."
        res = self.denken(frage)
        pattern = res["denkmuster"]
        trace = res.get("trace") or []
        focus_ids = res.get("focus") or []
        memory_hits = res.get("memories") or []

        if not pattern:
            return "Das weiß ich nicht. Bitte stelle mir eine neue Frage."

        if auto_lernen:
            if not (self.pred_enabled and self.pred_learning_only):
                self.lerne_aus_aktivierung(pattern, trace)

        use_lm_final = self.use_lm_default if use_lm is None else use_lm
        return self.versprachliche(
            pattern,
            intent=res["intent"],
            trace=trace,
            use_lm=use_lm_final,
            focus_ids=focus_ids,
            memory_hits=memory_hits,
        )

    def _is_greeting(self, text: str) -> bool:
        t = self._norm_label(text)
        if not t:
            return False
        greetings = {
            "hallo", "hi", "hey", "moin", "servus",
            "guten tag", "guten morgen", "guten abend",
        }
        if t in greetings:
            return True
        if t.startswith("guten "):
            return True
        return False

    def _use_embeddings_for(self, text: str) -> bool:
        toks = self._tokenize(text)
        if not toks:
            return False
        if len(toks) < 2 and len(text.strip()) < 6:
            return False
        return True

    def _focus_from_question(self, frage: str, seed_set: set[str]) -> List[str]:
        if not seed_set:
            return []
        tokens = self._tokenize(frage)
        if not tokens:
            return []
        stop = {
            "der", "die", "das", "ein", "eine", "einen", "einem", "einer",
            "ist", "sind", "und", "oder", "zu", "im", "in", "am", "an", "von", "mit",
            "fuer", "für", "den", "dem", "des", "hat", "haben", "besteht", "bestehen",
            "lebt", "gibt", "was", "wie", "warum", "wieso", "weshalb",
            "woraus", "womit", "wodurch", "wo", "wann", "wer", "wen", "wem", "wessen",
            "welche", "welcher", "welches", "welchen", "welchem",
            "außerdem", "ausserdem", "hierbei", "dabei", "somit", "jedoch",
        }
        tokens = [t for t in tokens if t not in stop]
        if not tokens:
            return []

        focus: List[str] = []
        used = set()

        # Prefer longest lexikon phrase matches in token order.
        i = 0
        while i < len(tokens):
            match = ""
            match_len = 0
            if self.lexikon:
                for ln in range(min(3, len(tokens) - i), 0, -1):
                    phrase = " ".join(tokens[i:i + ln])
                    cid = self.lexikon.get(phrase)
                    if cid and cid in seed_set:
                        match = cid
                        match_len = ln
                        break
            if match:
                if match not in used:
                    focus.append(match)
                    used.add(match)
                i += match_len
            else:
                i += 1

        if not focus and self.lexikon:
            for tok in tokens:
                cid = self.lexikon.get(tok)
                if cid and cid in seed_set and cid not in used:
                    focus.append(cid)
                    used.add(cid)

        return focus[:2]

    # -------------------------
    # Lernen (Hebb + Anti-Hebb + Decay)
    # -------------------------

    def _get_or_create_episodic_edge(
        self,
        src: str,
        dst: str,
        typ: str,
        w_init: float,
    ) -> Verbindung:
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
            e = self._get_or_create_episodic_edge(
                t.src,
                t.dst,
                self._canon_type(t.typ),
                w_init=w0,
            )
            boost = max(act_map.get(t.src, 0.0), act_map.get(t.dst, 0.0))
            e.gewicht = min(0.99, e.gewicht + self.lr_hebb * boost)

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
                        self._get_or_create_episodic_edge(
                            a,
                            c,
                            "assoziation",
                            w_init=self.triangle_create_w,
                        )

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

    def _get_or_create_semantic_edge(
        self,
        src: str,
        dst: str,
        typ: str,
        w_init: float,
    ) -> Verbindung:
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
                w_init = e.gewicht * self.consolidate_ratio
                se = self._get_or_create_semantic_edge(
                    src,
                    e.ziel,
                    self._canon_type(e.typ),
                    w_init=w_init,
                )
                se.gewicht = min(0.99, max(se.gewicht, w_init))
                se.gewicht = min(0.99, se.gewicht + self.consolidate_boost)

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
                obj = {
                    "t": "node",
                    "id": cid,
                    "features": sorted(set(feats)),
                    "labels": sorted(set(labels)),
                }
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
                    obj = {
                        "t": "node",
                        "id": cid,
                        "features": sorted(set(feats)),
                        "labels": sorted(set(labels)),
                    }
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

        msg = f"OK Modelle gespeichert: {datei}"
        if episodic_datei:
            msg = f"{msg} + {episodic_datei}"
        print(msg)

    def _speichere_txt(self, datei: str, episodic_datei: Optional[str] = None):
        with open(self._resolve_path(datei), "w", encoding="utf-8") as f:
            f.write("# KOGNITIVES MODELL (SEMANTIC) - AUTO-GENERIERT\n")
            f.write("# Format: konzept | features | verbindungen\n\n")
            for konzept_id, konzept in sorted(self.konzepte.items()):
                if konzept.semantische_features:
                    features = ",".join(konzept.semantische_features)
                else:
                    features = "gelernt"
                verbindungen = ", ".join(
                    [
                        f"{v.ziel}:{v.gewicht:.2f}:{self._canon_type(v.typ)}"
                        for v in konzept.verbindungen
                    ]
                )
                f.write(f"{konzept_id} | {features} | {verbindungen}\n")

        if episodic_datei:
            with open(self._resolve_path(episodic_datei), "w", encoding="utf-8") as f:
                f.write("# KOGNITIVES MODELL (EPISODIC) - AUTO-GENERIERT\n")
                f.write("# Format: konzept | features | verbindungen\n\n")
                for src, edges in sorted(self.episodic_edges.items()):
                    verbindungen = ", ".join(
                        [
                            f"{v.ziel}:{v.gewicht:.2f}:{self._canon_type(v.typ)}"
                            for v in edges
                        ]
                    )
                    f.write(f"{src} | episodic | {verbindungen}\n")

        msg = f"OK Modelle gespeichert: {datei}"
        if episodic_datei:
            msg = f"{msg} + {episodic_datei}"
        print(msg)

    # -------------------------
    # Autonomes Denken
    # -------------------------

    def autonom_denken(self, steps: int = 1) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        steps = max(1, min(steps, 10))

        for _ in range(steps):
            recent = self._think_recent if isinstance(self._think_recent, list) else []
            cooldown = max(0, int(self.think_cooldown))
            recent_set = set(recent[-cooldown:]) if cooldown > 0 else set()

            candidates = []
            goal_ids = self._goal_ids()
            for kid, k in self.konzepte.items():
                if self._is_junk_concept_id(kid):
                    continue
                deg = self._degree(kid)
                if deg <= 0:
                    continue
                sem_strength = sum(e.gewicht for e in k.verbindungen)
                epi_strength = sum(e.gewicht for e in self.episodic_edges.get(kid, []))
                novelty = 1.0 / (1.0 + sem_strength + epi_strength)
                gap = max(0.0, epi_strength - sem_strength)
                score = gap + 0.6 * novelty + random.uniform(0.0, 0.05)

                # Penalties: low-degree noise + cooldown to prevent repetition
                if deg <= 2:
                    score -= self.think_low_degree_penalty * (2 - deg + 1) * 0.5
                if kid in recent_set:
                    score *= 0.2

                candidates.append((score, kid, sem_strength, epi_strength, deg))

            candidates.sort(key=lambda x: x[0], reverse=True)

            def _weighted_pick(items, exclude=None, diversity_base=None) -> str:
                exclude = exclude or set()
                weights = []
                pool = []
                for s, kid, _, _, _ in items:
                    if kid in exclude:
                        continue
                    w = max(0.0, s)
                    if diversity_base and diversity_base(kid):
                        w *= self.think_link_penalty
                    if w <= 0.0:
                        continue
                    pool.append(kid)
                    weights.append(w)
                if not pool:
                    return ""
                total = sum(weights)
                r = random.random() * total
                acc = 0.0
                for kid, w in zip(pool, weights):
                    acc += w
                    if acc >= r:
                        return kid
                return pool[-1]

            topk = self.think_topk if self.think_topk and self.think_topk > 0 else len(candidates)
            top = candidates[:topk]

            seeds: List[str] = []
            for g in goal_ids[:2]:
                if g and g not in seeds:
                    seeds.append(g)

            if len(seeds) < 2:
                s1 = _weighted_pick(top, exclude=set(seeds))
                if s1 and s1 not in seeds:
                    seeds.append(s1)

            if len(seeds) < 2:
                def _is_linked(kid: str) -> bool:
                    return any(self._has_edge(kid, s) or self._has_edge(s, kid) for s in seeds)

                s2 = _weighted_pick(top, exclude=set(seeds), diversity_base=_is_linked)
                if s2 and s2 not in seeds:
                    seeds.append(s2)

            if not seeds:
                break

            cues = {seeds[0]: 0.95}
            if len(seeds) > 1:
                cues[seeds[1]] = 0.75

            if self.pred_enabled:
                act = self.predictive_activation(cues, intent="OTHER")
            else:
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
                if not (self.pred_enabled and self.pred_learning_only):
                    self.lerne_aus_aktivierung(pattern, res["trace"])
            results.append(res)

            # Update diversity memory
            for s in seeds:
                if s:
                    recent.append(s)
            if self.think_recent_max and len(recent) > self.think_recent_max:
                self._think_recent = recent[-self.think_recent_max:]
            else:
                self._think_recent = recent

        return results


# ============================================================
# ROWCOUNT FOOTER
# ============================================================
# ROWCOUNT_OLD: 664
# ROWCOUNT_NEW: 987
