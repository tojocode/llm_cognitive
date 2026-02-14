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
        self.cause_gate_match = 1.5
        self.cause_gate_mismatch = 0.3

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
        self.consolidate_score_threshold = 0.62
        self.consolidate_evidence_target = 2.8
        self.consolidate_min_confidence = 0.22

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

        # Value-System / Gating (Ziele)
        self.value_gate_enabled = True
        self.value_goal_boost = 0.6
        self.value_low_degree_penalty = 0.25

        # Frage-Anker (Warum)
        self.cause_focus_boost = 0.22
        self.question_anchor_boost = 0.12
        self.question_anchor_rank_boost = 0.35
        self.def_single_token_penalty = 0.85
        self.answer_min_relevance = 0.18
        self.answer_min_relevance_cause = 0.26
        self.value_generic_penalty = 0.28

        # Content/Context Trennung
        self.content_cue_weight = 1.0
        self.context_cue_weight = 0.42
        self.context_generic_penalty = 0.78

        # Schema-/Hierarchie-Layer
        self.schema_mode_enabled = True
        self.schema_core_relations = {
            "is_a",
            "part_of",
            "causes",
            "has_property",
            "located_in",
            "in_context",
            "related_to",
            "sequence",
        }
        self.generic_gate_penalty = 0.42
        self.generic_gate_focus_boost = 0.78

        # Planning (Lookahead)
        self.plan_enabled = True
        self.plan_width = 8
        self.plan_boost = 0.35
        self.plan_intent_bonus = 0.15

        # Cluster-First Memory (hierarchische Themenräume)
        self.cluster_mode_enabled = True
        self.cluster_lp_iters = 8
        self.cluster_min_size = 4
        self.cluster_core_edge_min_w = 0.14
        self.cluster_aug_edge_min_w = 0.28
        self.cluster_weak_types = {"cooccur", "coactive", "similar", "cluster_of"}
        self.cluster_bridge_types = {
            "ist",
            "hat",
            "teil_von",
            "klasse",
            "gehört_zu",
            "gehört_zu",
            "besteht_aus",
            "enthält",
            "enthält",
            "verursacht",
            "verursacht_durch",
            "lebt_in",
            "farbe",
            "eigenschaft",
        }
        self.cluster_bridge_min_w = 0.26
        self.cluster_bridge_max_per_node = 2
        self.cluster_bridge_strict = True
        self.cluster_local_bias = 1.08
        self.cluster_cross_bias = 0.72
        self.cluster_cue_cross_penalty = 0.86
        self.cluster_inhib_competitor = 0.24
        self.cluster_allow_top = 1

        self.cluster_of: Dict[str, str] = {}
        self.cluster_members: Dict[str, set[str]] = {}
        self.cluster_bridges: Dict[str, set[str]] = {}
        self.allowed_bridge_edges: set[tuple[str, str, str]] = set()
        self.active_clusters: set[str] = set()
        self._cluster_intent = "OTHER"

        # Ensemble Layer (multi-zugeordnete neuronale Gruppen)
        self.ensembles_enabled = True
        self.ensemble_min_size = 3
        self.ensemble_max_per_node = 4
        self.ensemble_boost = 0.18
        self.ensemble_context_boost = 0.08
        self.ensemble_coactive_lr = 0.04
        self.ensemble_rel_types = {"ist", "klasse", "gehört_zu", "gehört_zu", "teil_von", "farbe"}

        self.ensembles: Dict[str, Dict[str, float]] = {}
        self.node_ensembles: Dict[str, Dict[str, float]] = {}
        self.active_ensembles: set[str] = set()

        if self.semantic_datei:
            self.modell_laden(self.semantic_datei, episodic=False)
        if self.episodic_datei:
            self.modell_laden(self.episodic_datei, episodic=True)

        self._rebuild_cluster_index()
        self._rebuild_cluster_bridges()
        self._rebuild_ensembles()

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

    def _cluster_for(self, kid: str) -> str:
        if not kid:
            return ""
        c = self.cluster_of.get(kid, "")
        if c:
            return c
        k = self.konzepte.get(kid)
        if not k:
            return ""
        return (getattr(k, "cluster_id", "") or "").strip()

    def _is_cluster_candidate_node(self, kid: str) -> bool:
        if not kid:
            return False
        if kid.startswith("Cluster_"):
            return False
        if self._is_junk_concept_id(kid):
            return False
        return kid in self.konzepte

    def _iter_cluster_edges(self) -> Iterable[Tuple[str, str, float]]:
        for src, k in self.konzepte.items():
            if not self._is_cluster_candidate_node(src):
                continue
            for e in k.verbindungen:
                dst = e.ziel
                if not self._is_cluster_candidate_node(dst):
                    continue
                typ = self._canon_type(e.typ)
                w = float(e.gewicht or 0.0)
                if typ in self.cluster_weak_types:
                    if w < self.cluster_aug_edge_min_w:
                        continue
                    w *= 0.65
                else:
                    if w < self.cluster_core_edge_min_w:
                        continue
                if w <= 0.0:
                    continue
                yield src, dst, min(1.0, w)

    def _rebuild_cluster_index(self):
        self.cluster_of = {}
        self.cluster_members = {}

        if not self.cluster_mode_enabled or not self.konzepte:
            for k in self.konzepte.values():
                k.cluster_id = ""
            return

        nodes = [cid for cid in self.konzepte if self._is_cluster_candidate_node(cid)]
        if not nodes:
            for k in self.konzepte.values():
                k.cluster_id = ""
            return

        neighbors: Dict[str, Dict[str, float]] = {cid: {} for cid in nodes}
        for src, dst, w in self._iter_cluster_edges():
            if src == dst:
                continue
            prev = neighbors[src].get(dst, 0.0)
            if w > prev:
                neighbors[src][dst] = w
            prev_r = neighbors[dst].get(src, 0.0)
            if w > prev_r:
                neighbors[dst][src] = w

        labels = {cid: cid for cid in nodes}
        ordered_nodes = sorted(nodes)
        for _ in range(max(1, int(self.cluster_lp_iters))):
            changed = 0
            for cid in ordered_nodes:
                neigh = neighbors.get(cid) or {}
                if not neigh:
                    continue
                scores: Dict[str, float] = {}
                for nb, w in neigh.items():
                    lb = labels.get(nb, nb)
                    scores[lb] = scores.get(lb, 0.0) + float(w)
                if not scores:
                    continue
                best = sorted(scores.items(), key=lambda x: (x[1], x[0]), reverse=True)[0][0]
                if best != labels[cid]:
                    labels[cid] = best
                    changed += 1
            if changed == 0:
                break

        groups: Dict[str, List[str]] = {}
        for cid, lb in labels.items():
            groups.setdefault(lb, []).append(cid)

        large = {lb for lb, members in groups.items() if len(members) >= self.cluster_min_size}
        if large:
            for cid in ordered_nodes:
                lb = labels[cid]
                if lb in large:
                    continue
                neigh = neighbors.get(cid) or {}
                cand: Dict[str, float] = {}
                for nb, w in neigh.items():
                    nb_lb = labels.get(nb, "")
                    if nb_lb not in large:
                        continue
                    cand[nb_lb] = cand.get(nb_lb, 0.0) + float(w)
                if cand:
                    best_cand = sorted(
                        cand.items(),
                        key=lambda x: (x[1], x[0]),
                        reverse=True,
                    )[0][0]
                    labels[cid] = best_cand

        groups = {}
        for cid, lb in labels.items():
            groups.setdefault(lb, []).append(cid)

        sorted_groups = sorted(groups.values(), key=lambda g: (-len(g), min(g)))
        for i, members in enumerate(sorted_groups, 1):
            cluster_id = f"C{i:03d}"
            for cid in members:
                self.cluster_of[cid] = cluster_id
                self.konzepte[cid].cluster_id = cluster_id
                self.cluster_members.setdefault(cluster_id, set()).add(cid)

        for cid, k in self.konzepte.items():
            if cid in self.cluster_of:
                continue
            k.cluster_id = ""

    def _rebuild_cluster_bridges(self):
        self.cluster_bridges = {}
        self.allowed_bridge_edges = set()

        if not self.cluster_mode_enabled or not self.cluster_of:
            return

        pair_count: Dict[Tuple[str, str], int] = {}
        pair_weight: Dict[Tuple[str, str], float] = {}
        raw: Dict[str, List[Tuple[float, str, str]]] = {}

        for src, k in self.konzepte.items():
            c_src = self._cluster_for(src)
            if not c_src:
                continue
            for e in k.verbindungen:
                dst = e.ziel
                c_dst = self._cluster_for(dst)
                if not c_dst or c_dst == c_src:
                    continue
                typ = self._canon_type(e.typ)
                w = float(e.gewicht or 0.0)
                if typ not in self.cluster_bridge_types or w < self.cluster_bridge_min_w:
                    continue
                pair = tuple(sorted((c_src, c_dst)))
                pair_count[pair] = pair_count.get(pair, 0) + 1
                pair_weight[pair] = pair_weight.get(pair, 0.0) + w
                raw.setdefault(src, []).append((w, dst, typ))

        strong_pairs = {
            pair
            for pair, cnt in pair_count.items()
            if cnt >= 2 or pair_weight.get(pair, 0.0) >= 0.85
        }
        if not strong_pairs:
            strong_pairs = set(pair_count.keys())

        max_per_node = max(1, int(self.cluster_bridge_max_per_node))
        for src, cand in raw.items():
            c_src = self._cluster_for(src)
            if not c_src:
                continue
            cand.sort(key=lambda x: x[0], reverse=True)
            used_clusters: set[str] = set()
            kept = 0
            for w, dst, typ in cand:
                c_dst = self._cluster_for(dst)
                if not c_dst or c_dst == c_src:
                    continue
                pair = tuple(sorted((c_src, c_dst)))
                if pair not in strong_pairs:
                    continue
                if c_dst in used_clusters:
                    continue
                self.allowed_bridge_edges.add((src, dst, typ))
                self.cluster_bridges.setdefault(c_src, set()).add(c_dst)
                self.cluster_bridges.setdefault(c_dst, set()).add(c_src)
                used_clusters.add(c_dst)
                kept += 1
                if kept >= max_per_node:
                    break

    def _clear_cluster_context(self):
        self.active_clusters = set()
        self.active_ensembles = set()
        self._cluster_intent = "OTHER"

    def _set_active_clusters_from_cues(
        self,
        cues: Dict[str, float],
        anchors: List[str],
        intent: str,
    ):
        self.active_clusters = set()
        if not self.cluster_mode_enabled:
            return

        scores: Dict[str, float] = {}
        for kid, val in cues.items():
            c = self._cluster_for(kid)
            if not c:
                continue
            w = float(val)
            if self._is_generic_concept(kid):
                w *= 0.45
            scores[c] = scores.get(c, 0.0) + w

        anchor_noise = {"unterschied", "vergleich", "differenz", "abgrenzung"}
        for kid in anchors or []:
            c = self._cluster_for(kid)
            if not c:
                continue
            norm = self._norm_label(self._label_for_output(kid))
            toks = set(norm.split()) if norm else set()
            if toks.intersection(anchor_noise):
                continue
            boost = 0.8 if self._is_generic_concept(kid) else 1.2
            scores[c] = scores.get(c, 0.0) + boost

        if not scores:
            return

        max_clusters = 2 if intent == "COMPARE" else max(1, int(self.cluster_allow_top))
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_clusters]
        self.active_clusters = {c for c, s in ranked if s > 0}

    def _cluster_node_bias(self, kid: str, *, cue_mode: bool = False) -> float:
        if not self.cluster_mode_enabled or not self.active_clusters:
            return 1.0
        c = self._cluster_for(kid)
        if not c:
            return 1.0
        if c in self.active_clusters:
            return max(1.0, self.cluster_local_bias)
        if cue_mode:
            return max(0.05, min(1.0, self.cluster_cue_cross_penalty))
        return max(0.05, min(1.0, self.cluster_cross_bias))

    def _is_bridge_transition_allowed(
        self,
        src: str,
        dst: str,
        edge_typ: str,
        edge_w: float,
        c_src: str,
        c_dst: str,
    ) -> bool:
        if not self.cluster_bridge_strict:
            return True

        typ = self._canon_type(edge_typ)
        edge_key = (src, dst, typ)
        if edge_key in self.allowed_bridge_edges:
            return True

        if (
            self._cluster_intent == "COMPARE"
            and c_src in self.active_clusters
            and c_dst in self.active_clusters
        ):
            if typ in self.cluster_bridge_types and edge_w >= 0.20:
                return True

        if c_dst not in self.cluster_bridges.get(c_src, set()):
            return False

        if typ in self.cluster_bridge_types and edge_w >= self.cluster_bridge_min_w * 0.9:
            return True
        if typ not in self.cluster_weak_types and edge_w >= max(self.cluster_bridge_min_w, 0.34):
            return True
        return False

    def _cluster_transition_bias(
        self,
        src: str,
        dst: str,
        edge_typ: str,
        edge_w: float,
    ) -> float:
        if not self.cluster_mode_enabled or not self.active_clusters:
            return 1.0
        c_src = self._cluster_for(src)
        c_dst = self._cluster_for(dst)
        if not c_src or not c_dst:
            return 1.0

        if c_src == c_dst:
            if c_src in self.active_clusters:
                return max(1.0, self.cluster_local_bias)
            return 1.0

        if not self._is_bridge_transition_allowed(src, dst, edge_typ, edge_w, c_src, c_dst):
            return 0.0

        if c_src in self.active_clusters or c_dst in self.active_clusters:
            return max(0.05, min(1.0, self.cluster_cross_bias))
        return max(0.03, min(0.9, self.cluster_cross_bias * 0.9))

    def _ensure_node_ensemble(
        self,
        kid: str,
        ensemble_id: str,
        weight: float,
    ):
        if not kid or not ensemble_id:
            return
        self.node_ensembles.setdefault(kid, {})
        cur = self.node_ensembles[kid].get(ensemble_id, 0.0)
        self.node_ensembles[kid][ensemble_id] = max(cur, float(weight))

        self.ensembles.setdefault(ensemble_id, {})
        cur_rev = self.ensembles[ensemble_id].get(kid, 0.0)
        self.ensembles[ensemble_id][kid] = max(cur_rev, float(weight))

    def _rebuild_ensembles(self):
        self.ensembles = {}
        self.node_ensembles = {}
        if not self.ensembles_enabled:
            for k in self.konzepte.values():
                k.ensemble_ids = []
            return

        # Basis 1: Cluster-Ensembles
        for cluster_id, members in self.cluster_members.items():
            ens_id = f"ENS_CL_{cluster_id}"
            for kid in members:
                self._ensure_node_ensemble(kid, ens_id, 0.9)

        # Basis 2: Semantische Typ-Ensembles (multi assignment)
        for src, k in self.konzepte.items():
            if self._is_junk_concept_id(src):
                continue
            for e in k.verbindungen:
                typ = self._canon_type(e.typ)
                if typ not in self.ensemble_rel_types:
                    continue
                if float(e.gewicht or 0.0) < 0.18:
                    continue
                dst = e.ziel
                if not dst or self._is_junk_concept_id(dst):
                    continue
                ens_id = f"ENS_{typ}_{dst}"
                w_src = min(1.0, 0.42 + float(e.gewicht))
                self._ensure_node_ensemble(src, ens_id, w_src)
                self._ensure_node_ensemble(dst, ens_id, min(1.0, 0.34 + float(e.gewicht) * 0.6))

        # Entferne zu kleine Ensembles
        min_size = max(2, int(self.ensemble_min_size))
        valid_ensembles = {
            eid for eid, members in self.ensembles.items() if len(members) >= min_size
        }
        self.ensembles = {
            eid: members
            for eid, members in self.ensembles.items()
            if eid in valid_ensembles
        }

        pruned_node_ens: Dict[str, Dict[str, float]] = {}
        max_per_node = max(1, int(self.ensemble_max_per_node))
        for kid, memberships in self.node_ensembles.items():
            filtered = [(eid, w) for eid, w in memberships.items() if eid in valid_ensembles]
            filtered.sort(key=lambda x: x[1], reverse=True)
            top = filtered[:max_per_node]
            if top:
                pruned_node_ens[kid] = {eid: w for eid, w in top}
        self.node_ensembles = pruned_node_ens

        # Rückwärtsindex nach Pruning neu aufbauen
        new_ensembles: Dict[str, Dict[str, float]] = {}
        for kid, memberships in self.node_ensembles.items():
            for eid, w in memberships.items():
                new_ensembles.setdefault(eid, {})[kid] = w
        self.ensembles = new_ensembles

        # Persistente Felder in Konzepten
        for cid, k in self.konzepte.items():
            mids = self.node_ensembles.get(cid, {})
            k.ensemble_ids = sorted(mids.keys())

    def _select_active_ensembles(self, content_cues: Dict[str, float]):
        self.active_ensembles = set()
        if not self.ensembles_enabled or not content_cues:
            return

        scores: Dict[str, float] = {}
        for kid, val in sorted(content_cues.items(), key=lambda x: x[1], reverse=True)[:6]:
            memberships = self.node_ensembles.get(kid, {})
            for eid, w in memberships.items():
                scores[eid] = scores.get(eid, 0.0) + float(val) * float(w)

        if not scores:
            return

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:4]
        self.active_ensembles = {eid for eid, score in ranked if score > 0.05}

    def _apply_ensemble_bias_to_cues(self, cues: Dict[str, float]) -> Dict[str, float]:
        if not self.ensembles_enabled or not self.active_ensembles or not cues:
            return cues

        out: Dict[str, float] = {}
        for kid, val in cues.items():
            memberships = self.node_ensembles.get(kid, {})
            overlap = 0.0
            for eid in self.active_ensembles:
                overlap += float(memberships.get(eid, 0.0))
            boost = 1.0 + min(0.45, overlap * self.ensemble_boost)
            if self._is_generic_concept(kid):
                boost = 1.0 + min(0.2, overlap * self.ensemble_context_boost)
            out[kid] = min(0.99, float(val) * boost)
        return out

    def _learn_ensemble_coactivation(self, pattern: List[Tuple[str, float]]):
        if not self.ensembles_enabled or not pattern:
            return
        top = [(kid, act) for kid, act in pattern[:10] if act >= self.pattern_threshold]
        if len(top) < 2:
            return

        for src, src_act in top:
            src_ens = self.node_ensembles.get(src, {})
            if not src_ens:
                continue
            for dst, dst_act in top:
                if src == dst:
                    continue
                lr = self.ensemble_coactive_lr * min(float(src_act), float(dst_act))
                if lr <= 0:
                    continue
                cur = self.node_ensembles.get(dst, {}).copy()
                for eid, w in src_ens.items():
                    new_w = max(cur.get(eid, 0.0), min(1.0, float(w) + lr))
                    cur[eid] = new_w
                if cur:
                    keep = sorted(cur.items(), key=lambda x: x[1], reverse=True)[
                        : max(1, int(self.ensemble_max_per_node))
                    ]
                    self.node_ensembles[dst] = {eid: w for eid, w in keep}

        # Rückwärtsindex aktualisieren
        rebuilt: Dict[str, Dict[str, float]] = {}
        for kid, memberships in self.node_ensembles.items():
            for eid, w in memberships.items():
                rebuilt.setdefault(eid, {})[kid] = w
            if kid in self.konzepte:
                self.konzepte[kid].ensemble_ids = sorted(memberships.keys())
        self.ensembles = rebuilt

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

    def _schema_rel(self, rel_type: str) -> str:
        t = self._canon_type(rel_type).strip().lower()
        mapping = {
            "ist": "is_a",
            "klasse": "is_a",
            "gehört_zu": "is_a",
            "gehört_zu": "is_a",
            "gehoert_zu": "is_a",
            "gilt_als": "is_a",
            "teil_von": "part_of",
            "besteht_aus": "part_of",
            "enthält": "part_of",
            "enthält": "part_of",
            "enthaelt": "part_of",
            "hat": "has_property",
            "eigenschaft": "has_property",
            "eigenschaft_von": "has_property",
            "farbe": "has_property",
            "verursacht": "causes",
            "verursacht_durch": "causes",
            "verursacht_von": "causes",
            "durch": "causes",
            "ermöglicht": "causes",
            "lebt_in": "located_in",
            "in": "located_in",
            "von": "located_in",
            "folge": "sequence",
            "prozess": "in_context",
            "cooccur": "related_to",
            "coactive": "related_to",
            "similar": "related_to",
            "cluster_of": "related_to",
            "assoziation": "related_to",
        }
        schema = mapping.get(t)
        if schema is None:
            if t.startswith("cluster") or t.startswith("co"):
                schema = "related_to"
            else:
                schema = "in_context"

        # Nur freigegebene Kernrelationen zulassen; alles andere auf "related_to" mappen.
        if schema in self.schema_core_relations:
            return schema
        return "related_to"

    # -------------------------
    # Intent / Gating
    # -------------------------

    def _erkenne_intent(self, frage: str) -> str:
        f = frage.strip().lower()
        if f.startswith(("warum", "weshalb", "wieso", "wodurch", "womit")):
            return "CAUSE"
        if (
            f.startswith("worin unterscheidet sich")
            or "unterschied zwischen" in f
            or f.startswith("was ist der unterschied")
            or "vergleich" in f
        ):
            return "COMPARE"
        if f.startswith(("zu welchem", "zu welcher", "zu welchem system", "zu welcher klasse")):
            return "PARTS"
        if f.startswith(("wo ", "wohin", "woher")):
            return "WHERE"
        if f.startswith(("woraus", "woraus besteht", "woraus setzt", "woraus besteht")):
            return "PARTS"
        if f.startswith(
            (
                "welche eigenschaften",
                "welche merkmale",
                "welche eigenschaft",
                "welche farbe",
                "welche färbung",
                "welche faerbung",
            )
        ):
            return "PROPS"
        if f.startswith("wie"):
            return "HOW"
        if f.startswith("was"):
            return "DEF"
        if f.startswith(("ist ", "sind ", "hat ", "haben ")):
            return "DEF"
        return "OTHER"

    def _gate(self, edge_type: str, intent: str) -> float:
        schema = self._schema_rel(edge_type)

        if intent == "DEF":
            preferred = {"is_a", "part_of"}
        elif intent == "CAUSE":
            preferred = {"causes"}
        elif intent == "WHERE":
            preferred = {"located_in"}
        elif intent == "PARTS":
            preferred = {"part_of", "is_a"}
        elif intent == "PROPS":
            preferred = {"has_property", "is_a"}
        elif intent == "HOW":
            preferred = {"causes", "part_of", "sequence", "in_context"}
        elif intent == "COMPARE":
            preferred = {"is_a", "has_property", "part_of"}
        else:
            preferred = set()

        if intent == "CAUSE":
            return self.cause_gate_match if schema in preferred else self.cause_gate_mismatch

        if intent == "PROPS":
            if schema == "has_property":
                return max(self.gate_match, 1.35)
            if schema == "is_a":
                return max(self.gate_mismatch, 0.58)
            if schema == "part_of":
                return max(self.gate_mismatch, 0.5)
            return self.gate_mismatch

        if schema in preferred:
            return self.gate_match
        if self.schema_mode_enabled and schema == "related_to":
            return min(self.gate_mismatch, 0.52)
        return self.gate_mismatch

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
                        cluster_id = self._nfc(str(obj.get("cluster", ""))).strip()
                        context_ids = obj.get("contexts") or []
                        context_ids = [self._nfc(str(x)) for x in context_ids if str(x).strip()]
                        ensemble_ids = obj.get("ensembles") or []
                        ensemble_ids = [self._nfc(str(x)) for x in ensemble_ids if str(x).strip()]
                        if not labels:
                            labels = [cid]
                        if cid not in self.konzepte:
                            self.konzepte[cid] = Konzept(
                                id=cid,
                                labels=labels,
                                semantische_features=feats,
                                cluster_id=cluster_id,
                                context_ids=context_ids,
                                ensemble_ids=ensemble_ids,
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
                            if context_ids:
                                ex_ctx = set(self.konzepte[cid].context_ids or [])
                                for ctx in context_ids:
                                    if ctx not in ex_ctx:
                                        self.konzepte[cid].context_ids.append(ctx)
                                        ex_ctx.add(ctx)
                            if ensemble_ids:
                                ex_ens = set(self.konzepte[cid].ensemble_ids or [])
                                for ens in ensemble_ids:
                                    if ens not in ex_ens:
                                        self.konzepte[cid].ensemble_ids.append(ens)
                                        ex_ens.add(ens)
                            if cluster_id:
                                self.konzepte[cid].cluster_id = cluster_id
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
                        try:
                            evidence = float(obj.get("evidence", 0.0))
                        except Exception:
                            evidence = 0.0
                        try:
                            confidence = float(obj.get("confidence", 0.0))
                        except Exception:
                            confidence = 0.0
                        try:
                            context_stability = float(obj.get("context", 0.0))
                        except Exception:
                            context_stability = 0.0
                        if src not in self.konzepte:
                            self.konzepte[src] = Konzept(id=src, labels=[src])
                        if dst not in self.konzepte:
                            self.konzepte[dst] = Konzept(id=dst, labels=[dst])
                        edge = Verbindung(
                            ziel=dst,
                            gewicht=w,
                            typ=typ,
                            evidence=max(0.0, evidence),
                            confidence=max(0.0, min(1.0, confidence)),
                            context_stability=max(0.0, min(1.0, context_stability)),
                        )
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

        if self.cluster_mode_enabled and self.cluster_inhib_competitor > 0 and out:
            c_scores: Dict[str, float] = {}
            for kid, val in out.items():
                c = self._cluster_for(kid)
                if not c:
                    continue
                c_scores[c] = c_scores.get(c, 0.0) + float(val)

            if c_scores:
                ranked = sorted(c_scores.items(), key=lambda x: x[1], reverse=True)
                if self.active_clusters:
                    active_ranked = [(c, s) for c, s in ranked if c in self.active_clusters]
                    if active_ranked:
                        ranked = active_ranked
                keep_n = (
                    2
                    if self._cluster_intent == "COMPARE"
                    else max(1, int(self.cluster_allow_top))
                )
                keep = {c for c, _ in ranked[:keep_n]}

                if keep:
                    damp = max(0.0, min(0.9, float(self.cluster_inhib_competitor)))
                    for kid, val in list(out.items()):
                        c = self._cluster_for(kid)
                        if not c or c in keep:
                            continue
                        out[kid] = max(0.0, val * (1.0 - damp))

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
                if self._is_junk_concept_id(e.ziel):
                    continue
                yield src, e, "semantic"
        if src in self.episodic_edges:
            for e in self.episodic_edges[src]:
                if not include_seq and e.typ == self.seq_type:
                    continue
                if self._is_junk_concept_id(e.ziel):
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
                    incoming *= self._cluster_transition_bias(src, dst, e.typ, e.gewicht)

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
                    incoming *= self._cluster_transition_bias(src, dst, e.typ, e.gewicht)

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
        self._clear_cluster_context()
        self._cluster_intent = intent
        if self._is_greeting(frage):
            return {
                "frage": frage,
                "intent": intent,
                "unknown": "",
                "denkmuster": [],
                "focus": [],
                "anchors": [],
                "memories": [],
                "trace": [],
                "smalltalk": "greeting",
                "timestamp": datetime.now().isoformat(),
            }

        content_tokens, context_tokens = self._split_question_tokens(frage)

        base_raw_cues = self._cue_set(frage)
        base_content_cues, base_context_cues = self._split_cues_content_context(
            base_raw_cues,
            content_tokens,
            context_tokens,
        )

        content_cues = dict(base_content_cues)
        context_cues = dict(base_context_cues)
        memory_hits: List[Dict[str, object]] = []
        question_anchors = self._anchor_from_question(frage)

        for kid in question_anchors:
            toks = self._concept_query_tokens(kid)
            overlap_content = len(toks.intersection(content_tokens)) if content_tokens else 0
            if overlap_content > 0 or kid in content_cues:
                content_cues[kid] = max(
                    content_cues.get(kid, 0.0),
                    0.95 + self.question_anchor_boost,
                )
            else:
                context_cues[kid] = max(
                    context_cues.get(kid, 0.0),
                    0.72 + self.question_anchor_boost,
                )

        anchored: List[str] = []
        if intent == "CAUSE":
            anchored = question_anchors
            for kid in anchored:
                content_cues[kid] = max(content_cues.get(kid, 0.0), 0.95 + self.cause_focus_boost)

        cues = self._merge_content_context_cues(content_cues, context_cues)
        seed_set = set(content_cues.keys()) if content_cues else set(cues.keys())
        focus_ids = [k for k, _ in sorted(cues.items(), key=lambda x: x[1], reverse=True)[:2]]

        if cues:
            cues = {k: v for k, v in cues.items() if not self._is_junk_concept_id(k)}
            seed_set = {k for k in seed_set if k in cues}

        # Hybrid memory: retrieve similar texts and use as weak cues
        allow_embed = True
        embed_strong = self.embed_min_score_strong
        if intent == "CAUSE":
            if not base_content_cues and not anchored:
                allow_embed = False
            embed_strong = max(embed_strong, self.embed_min_score_strong * 1.4)

        if allow_embed and self.embed_enabled and self._use_embeddings_for(frage):
            memory_hits = self.embed_query(frage)
            top_score = float(memory_hits[0].get("score") or 0.0) if memory_hits else 0.0
            if not base_content_cues and top_score < embed_strong:
                memory_hits = []
            for hit in memory_hits:
                score = float(hit.get("score") or 0.0)
                if score <= 0:
                    continue
                mcues = self._cue_set(str(hit.get("text") or ""))
                m_content, m_context = self._split_cues_content_context(
                    mcues,
                    content_tokens,
                    context_tokens,
                )
                boost = self.embed_cue_boost * score
                for k, v in m_content.items():
                    content_cues[k] = max(content_cues.get(k, 0.0), min(0.99, v * boost))
                for k, v in m_context.items():
                    context_cues[k] = max(context_cues.get(k, 0.0), min(0.99, v * boost * 0.9))

        # Goal Layer cues (soft bias)
        for k, v in self._goal_cues().items():
            content_cues[k] = max(content_cues.get(k, 0.0), v)

        cues = self._merge_content_context_cues(content_cues, context_cues)
        seed_set = set(content_cues.keys()) if content_cues else set(cues.keys())

        self._select_active_ensembles(content_cues)
        cues = self._apply_ensemble_bias_to_cues(cues)

        cluster_seed_cues = content_cues if content_cues else cues
        self._set_active_clusters_from_cues(cluster_seed_cues, question_anchors, intent)
        if self.active_clusters and cues:
            biased: Dict[str, float] = {}
            for kid, val in cues.items():
                fac = self._cluster_node_bias(kid, cue_mode=True)
                v2 = float(val) * fac
                if v2 > self.cutoff * 0.5:
                    biased[kid] = v2
            if biased:
                cues = self._apply_ensemble_bias_to_cues(biased)
                seed_set = {k for k in seed_set if k in cues}

        if not cues:
            unknown = self._create_unknown_stub(frage)
            return {
                "frage": frage,
                "intent": intent,
                "unknown": unknown,
                "denkmuster": [],
                "focus": focus_ids,
                "anchors": question_anchors,
                "content_tokens": sorted(content_tokens),
                "context_tokens": sorted(context_tokens),
                "content_cues": sorted(content_cues.keys())[:8],
                "context_cues": sorted(context_cues.keys())[:8],
                "active_ensembles": sorted(self.active_ensembles),
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
            [
                (k, v)
                for k, v in act.items()
                if v >= self.pattern_threshold and not self._is_junk_concept_id(k)
            ],
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

        # Value-Gating: sortiere nach Frage-/Ziel-Passung
        goal_tokens = self._goal_tokens(frage)
        if self.value_gate_enabled and goal_tokens:
            scored = []
            anchor_set = set(question_anchors)
            for kid, val in pattern:
                v = self._value_score(kid, goal_tokens)
                bonus = self._plan_bonus(kid, goal_tokens, intent) if self.plan_enabled else 0.0
                score = val * v * (1.0 + bonus)
                score *= self._cluster_node_bias(kid, cue_mode=False)
                if kid in anchor_set:
                    score *= (1.0 + self.question_anchor_rank_boost)
                if (
                    intent == "DEF"
                    and len(goal_tokens) >= 2
                    and self._best_label_token_len(kid) <= 1
                ):
                    score *= self.def_single_token_penalty
                scored.append((kid, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            pattern = scored
        elif self.active_clusters and pattern:
            pattern = [
                (kid, val * self._cluster_node_bias(kid, cue_mode=False))
                for kid, val in pattern
            ]
            pattern.sort(key=lambda x: x[1], reverse=True)

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

        pattern = self._apply_generic_gate(pattern, focus_ids, intent)
        self._update_context_tags(pattern, context_tokens, intent)

        trace_out = sorted(self.trace, key=lambda t: t.contrib, reverse=True)[:20]
        if question_anchors:
            trace_srcs = {t.src for t in trace_out if getattr(t, "src", None)}
            for anchor in question_anchors:
                if anchor in trace_srcs:
                    continue
                trace_out.insert(
                    0,
                    TraceItem(
                        tick=0,
                        src=anchor,
                        dst=anchor,
                        typ="focus",
                        contrib=1.0,
                        layer="anchor",
                    ),
                )
                break
            trace_out = trace_out[:20]

        return {
            "frage": frage,
            "intent": intent,
            "unknown": "",
            "denkmuster": pattern,
            "focus": focus_ids,
            "anchors": question_anchors,
            "content_tokens": sorted(content_tokens),
            "context_tokens": sorted(context_tokens),
            "content_cues": sorted(content_cues.keys())[:8],
            "context_cues": sorted(context_cues.keys())[:8],
            "active_ensembles": sorted(self.active_ensembles),
            "memories": memory_hits,
            "trace": trace_out,
            "timestamp": datetime.now().isoformat(),
        }

    def _is_answer_confident(self, frage: str, res: Dict) -> bool:
        pattern = res.get("denkmuster") or []
        if not pattern:
            return False
        goal_tokens = self._goal_tokens(frage)
        if not goal_tokens:
            return True
        intent = str(res.get("intent") or "OTHER")
        focus_ids = res.get("focus") or []
        anchors = res.get("anchors") or []

        top_ids = [kid for kid, _ in pattern[:5]]
        rel_top = 0.0
        for kid in top_ids:
            rel_top = max(rel_top, self._label_match_ratio(kid, goal_tokens))

        rel_focus = 0.0
        for kid in focus_ids:
            rel_focus = max(rel_focus, self._label_match_ratio(str(kid), goal_tokens))

        rel_anchor = 0.0
        for kid in anchors:
            rel_anchor = max(rel_anchor, self._label_match_ratio(str(kid), goal_tokens))

        if intent == "CAUSE":
            if not anchors and rel_top < self.answer_min_relevance_cause:
                return False
            return max(rel_top, rel_focus, rel_anchor) >= self.answer_min_relevance_cause

        return max(rel_top, rel_focus, rel_anchor) >= self.answer_min_relevance

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

        if not self._is_answer_confident(frage, res):
            return "Das weiß ich nicht. Bitte stelle mir eine neue Frage."

        if auto_lernen:
            if self.pred_enabled and self.pred_learning_only:
                self.lerne_episodisch_trace(pattern, trace)
            else:
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

    def _context_terms(self) -> set[str]:
        return {
            "der", "die", "das", "ein", "eine", "einen", "einem", "einer",
            "ist", "sind", "und", "oder", "zu", "im", "in", "am", "an", "von", "mit",
            "fuer", "für", "den", "dem", "des", "hat", "haben", "besteht", "bestehen",
            "lebt", "gibt", "was", "wie", "warum", "wieso", "weshalb",
            "woraus", "womit", "wodurch", "wo", "wann", "wer", "wen", "wem", "wessen",
            "welche", "welcher", "welches", "welchen", "welchem",
            "außerdem", "ausserdem", "hierbei", "dabei", "somit", "jedoch",
            "sich", "unterschied", "vergleich", "differenz", "abgrenzung",
            "system", "wissenschaft", "modell", "prinzip",
            "wirkung", "auswirkung", "ursache", "folgen", "funktion", "prozess",
            "frage", "kontext", "inhalt", "warumfrage", "definitionsfrage",
            "zwischen", "gehört", "gehoert", "passt", "unterscheidet",
            "vergleichbar", "zuordnung", "kategorie",
        }

    def _split_question_tokens(self, frage: str) -> tuple[set[str], set[str]]:
        tokens = self._tokenize(frage)
        if not tokens:
            return set(), set()
        context_terms = self._context_terms()
        content: set[str] = set()
        context: set[str] = set()
        for t in tokens:
            if t in context_terms:
                context.add(t)
                continue
            if len(t) < 3 and not t.isupper():
                context.add(t)
                continue
            content.add(t)
        if not content:
            for t in tokens:
                if t not in context_terms and len(t) >= 2:
                    content.add(t)
                    break
        return content, context

    def _concept_query_tokens(self, kid: str) -> set[str]:
        if not kid:
            return set()
        k = self.konzepte.get(kid)
        labels = (k.labels if k and k.labels else [kid])
        out: set[str] = set()
        for lab in labels:
            norm = self._norm_label(lab)
            if not norm:
                continue
            for t in norm.split():
                if t:
                    out.add(t)
                    for var in self._token_variants(t):
                        if var:
                            out.add(var)
        return out

    def _split_cues_content_context(
        self,
        cues: Dict[str, float],
        content_tokens: set[str],
        context_tokens: set[str],
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        content_cues: Dict[str, float] = {}
        context_cues: Dict[str, float] = {}

        for kid, val in cues.items():
            if self._is_junk_concept_id(kid):
                continue
            toks = self._concept_query_tokens(kid)
            overlap_content = len(toks.intersection(content_tokens)) if content_tokens else 0
            overlap_context = len(toks.intersection(context_tokens)) if context_tokens else 0

            if overlap_content > 0:
                score = float(val) * (1.0 + 0.12 * overlap_content)
                if self._is_generic_concept(kid):
                    score *= 0.85
                content_cues[kid] = max(content_cues.get(kid, 0.0), score)
                continue

            if overlap_context > 0:
                score = float(val) * (0.78 if not self._is_generic_concept(kid) else 0.62)
                context_cues[kid] = max(context_cues.get(kid, 0.0), score)
                continue

            if content_tokens:
                # Strikte Trennung: ohne Inhalts- oder Kontextmatch keine Übernahme.
                continue

            if self._is_generic_concept(kid):
                context_cues[kid] = max(context_cues.get(kid, 0.0), float(val) * 0.42)
            else:
                content_cues[kid] = max(content_cues.get(kid, 0.0), float(val) * 0.68)

        return content_cues, context_cues

    def _merge_content_context_cues(
        self,
        content_cues: Dict[str, float],
        context_cues: Dict[str, float],
    ) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        for kid, val in content_cues.items():
            merged[kid] = max(merged.get(kid, 0.0), min(0.99, float(val) * self.content_cue_weight))
        for kid, val in context_cues.items():
            score = float(val) * self.context_cue_weight
            if self._is_generic_concept(kid):
                score *= (1.0 - self.context_generic_penalty)
            merged[kid] = max(merged.get(kid, 0.0), min(0.99, score))
        return merged

    def _update_context_tags(
        self,
        pattern: List[Tuple[str, float]],
        context_tokens: set[str],
        intent: str,
    ):
        if not pattern:
            return
        tags = [f"intent:{(intent or 'other').lower()}"]
        for tok in sorted(context_tokens):
            if len(tags) >= 5:
                break
            tags.append(f"ctx:{tok}")

        for kid, _score in pattern[:6]:
            k = self.konzepte.get(kid)
            if not k:
                continue
            existing = list(k.context_ids or [])
            for tag in tags:
                if tag not in existing:
                    existing.append(tag)
            if len(existing) > 10:
                existing = existing[-10:]
            k.context_ids = existing

    def _goal_tokens(self, frage: str) -> set[str]:
        content_tokens, _context_tokens = self._split_question_tokens(frage)
        return content_tokens

    def _value_score(self, kid: str, goal_tokens: set[str]) -> float:
        if not kid:
            return 0.5
        if self._is_junk_concept_id(kid):
            return 0.3
        if not goal_tokens:
            return 1.0
        k = self.konzepte.get(kid)
        labels = (k.labels if k and k.labels else [kid])
        best = 0.0
        for lab in labels:
            norm = self._norm_label(lab)
            if not norm:
                continue
            toks = set(norm.split())
            if not toks:
                continue
            overlap = len(goal_tokens.intersection(toks))
            best = max(best, overlap / max(1, len(goal_tokens)))
        value = 1.0 + self.value_goal_boost * best
        if self._degree(kid) <= 1:
            value *= (1.0 - self.value_low_degree_penalty)
        if self._is_generic_concept(kid):
            value *= (1.0 - self.value_generic_penalty)
        return max(0.5, min(1.8, value))

    def _is_generic_concept(self, kid: str) -> bool:
        generic = {
            "system",
            "wissenschaft",
            "begriff",
            "prinzip",
            "prozess",
            "modell",
            "struktur",
            "theorie",
            "methode",
            "form",
            "art",
        }
        if not kid:
            return False
        k = self.konzepte.get(kid)
        labels = (k.labels if k and k.labels else [kid])
        for lab in labels:
            norm = self._norm_label(lab)
            if not norm:
                continue
            toks = [t for t in norm.split() if t]
            if len(toks) != 1:
                return False
            if toks[0] not in generic:
                return False
        return True

    def _intent_rel_types(self, intent: str) -> set[str]:
        if intent == "DEF":
            return {"is_a", "part_of"}
        if intent == "CAUSE":
            return {"causes"}
        if intent == "PARTS":
            return {"part_of", "is_a"}
        if intent == "WHERE":
            return {"located_in", "is_a"}
        if intent == "PROPS":
            return {"has_property", "is_a"}
        if intent == "HOW":
            return {"causes", "part_of", "sequence", "in_context"}
        if intent == "COMPARE":
            return {"is_a", "has_property", "part_of"}
        return set()

    def _label_match_ratio(self, kid: str, goal_tokens: set[str]) -> float:
        if not goal_tokens:
            return 0.0
        k = self.konzepte.get(kid)
        labels = (k.labels if k and k.labels else [kid])
        best = 0.0
        for lab in labels:
            norm = self._norm_label(lab)
            if not norm:
                continue
            toks = set(norm.split())
            if not toks:
                continue
            overlap = len(goal_tokens.intersection(toks))
            best = max(best, overlap / max(1, len(goal_tokens)))
        return best

    def _plan_bonus(self, kid: str, goal_tokens: set[str], intent: str) -> float:
        if not goal_tokens:
            return 0.0
        rel_types = self._intent_rel_types(intent)
        bonus = 0.0
        edges = list(self._iter_edges(kid, include_seq=False))
        if not edges:
            return 0.0
        edges.sort(key=lambda x: x[1].gewicht, reverse=True)
        for _, e, _layer in edges[: max(1, int(self.plan_width))]:
            dst = e.ziel
            if not dst:
                continue
            if self._label_match_ratio(dst, goal_tokens) > 0:
                bonus += self.plan_boost * float(e.gewicht)
            e_schema = self._schema_rel(e.typ)
            if rel_types and e_schema in rel_types:
                bonus += self.plan_intent_bonus * float(e.gewicht)
        return min(0.6, bonus)

    def _apply_generic_gate(
        self,
        pattern: List[Tuple[str, float]],
        focus_ids: List[str],
        intent: str,
    ) -> List[Tuple[str, float]]:
        if not pattern:
            return pattern
        focus_set = set(focus_ids or [])
        out: List[Tuple[str, float]] = []

        for kid, val in pattern:
            score = float(val)
            if not self._is_generic_concept(kid):
                out.append((kid, score))
                continue

            factor = self.generic_gate_penalty
            if kid in focus_set:
                factor = max(factor, self.generic_gate_focus_boost)

            has_semantic_anchor = False
            for _, e, _layer in self._iter_edges(kid, include_seq=False):
                if e.ziel not in focus_set:
                    continue
                schema = self._schema_rel(e.typ)
                if (
                    schema in {"is_a", "part_of", "has_property", "located_in"}
                    and e.gewicht >= 0.32
                ):
                    has_semantic_anchor = True
                    break
            if has_semantic_anchor:
                factor = max(factor, 0.9 if intent in {"COMPARE", "PARTS"} else 0.82)

            out.append((kid, score * factor))

        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def _focus_from_question(self, frage: str, seed_set: set[str]) -> List[str]:
        if not seed_set:
            return []
        tokens = self._tokenize(frage)
        if not tokens:
            return []
        stop = self._context_terms()
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
                    if cid and cid in seed_set and not self._is_junk_concept_id(cid):
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
                if (
                    cid
                    and cid in seed_set
                    and cid not in used
                    and not self._is_junk_concept_id(cid)
                ):
                    focus.append(cid)
                    used.add(cid)
                    continue
                for alt in self._token_variants(tok):
                    if alt == tok:
                        continue
                    cid = self.lexikon.get(alt)
                    if (
                        cid
                        and cid in seed_set
                        and cid not in used
                        and not self._is_junk_concept_id(cid)
                    ):
                        focus.append(cid)
                        used.add(cid)
                        break

        return focus[:2]

    def _anchor_from_question(self, frage: str) -> List[str]:
        tokens = self._tokenize(frage)
        if not tokens or not self.lexikon:
            return []
        stop = self._context_terms()
        tokens = [t for t in tokens if t not in stop]
        if not tokens:
            return []

        focus: List[str] = []
        used = set()

        i = 0
        while i < len(tokens):
            match = ""
            match_len = 0
            for ln in range(min(3, len(tokens) - i), 0, -1):
                phrase = " ".join(tokens[i:i + ln])
                cid = self.lexikon.get(phrase)
                if cid and not self._is_junk_concept_id(cid):
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

        if not focus:
            for tok in tokens:
                cid = self.lexikon.get(tok)
                if cid and cid not in used and not self._is_junk_concept_id(cid):
                    focus.append(cid)
                    used.add(cid)
                    continue
                for alt in self._token_variants(tok):
                    if alt == tok:
                        continue
                    cid = self.lexikon.get(alt)
                    if cid and cid not in used and not self._is_junk_concept_id(cid):
                        focus.append(cid)
                        used.add(cid)
                        break

        return focus[:2]

    def _best_label_token_len(self, kid: str) -> int:
        if not kid:
            return 1
        k = self.konzepte.get(kid)
        labels = (k.labels if k and k.labels else [kid])
        best = 0
        for lab in labels:
            norm = self._norm_label(lab)
            if not norm:
                continue
            toks = [t for t in norm.split() if t]
            best = max(best, len(toks))
        return best if best > 0 else 1

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
        e = Verbindung(
            ziel=dst,
            gewicht=max(0.01, min(w_init, 0.99)),
            typ=typ,
            evidence=0.12,
            confidence=0.18,
            context_stability=self._edge_context_stability(src, dst),
        )
        self.episodic_edges[src].append(e)
        return e

    def _edge_context_stability(self, src: str, dst: str) -> float:
        score = 0.0
        c_src = self._cluster_for(src)
        c_dst = self._cluster_for(dst)
        if c_src and c_dst:
            if c_src == c_dst:
                score = max(score, 0.92)
            elif c_dst in self.cluster_bridges.get(c_src, set()):
                score = max(score, 0.62)
            elif self.active_clusters and (
                c_src in self.active_clusters or c_dst in self.active_clusters
            ):
                score = max(score, 0.4)

        ens_src = self.node_ensembles.get(src, {})
        ens_dst = self.node_ensembles.get(dst, {})
        shared = set(ens_src).intersection(set(ens_dst))
        if shared:
            shared_w = sum(min(float(ens_src[eid]), float(ens_dst[eid])) for eid in shared)
            score = max(score, min(0.95, 0.58 + 0.18 * shared_w))
        if self.active_ensembles and shared.intersection(self.active_ensembles):
            score = min(1.0, max(score, 0.76))

        ctx_src = set((self.konzepte.get(src).context_ids if self.konzepte.get(src) else []) or [])
        ctx_dst = set((self.konzepte.get(dst).context_ids if self.konzepte.get(dst) else []) or [])
        if ctx_src and ctx_dst and ctx_src.intersection(ctx_dst):
            score = min(1.0, max(score, 0.66))

        return max(0.0, min(1.0, score))

    def _consolidation_score(self, e: Verbindung) -> float:
        w = max(0.0, min(1.0, float(e.gewicht)))
        evidence = max(0.0, float(getattr(e, "evidence", 0.0) or 0.0))
        ev_norm = min(1.0, evidence / max(0.1, self.consolidate_evidence_target))
        conf = max(0.0, min(1.0, float(getattr(e, "confidence", 0.0) or 0.0)))
        ctx = max(0.0, min(1.0, float(getattr(e, "context_stability", 0.0) or 0.0)))
        return 0.35 * w + 0.30 * ev_norm + 0.20 * conf + 0.15 * ctx

    def lerne_episodisch_trace(self, pattern: List[Tuple[str, float]], trace: List[TraceItem]):
        act_map = {k: a for k, a in pattern}

        for t in trace[:12]:
            if t.contrib <= 0:
                continue
            e = self._get_or_create_episodic_edge(
                t.src,
                t.dst,
                self._canon_type(t.typ),
                w_init=0.2 + min(0.45, t.contrib),
            )
            boost = max(act_map.get(t.src, 0.0), act_map.get(t.dst, 0.0))
            e.gewicht = min(0.99, e.gewicht + self.lr_hebb * 0.45 * boost)

            e.evidence = min(999.0, float(getattr(e, "evidence", 0.0) or 0.0) + 0.25 + t.contrib)
            obs_conf = max(0.0, min(1.0, float(t.contrib) * 1.05))
            e.confidence = max(
                obs_conf,
                0.9 * float(getattr(e, "confidence", 0.0) or 0.0) + 0.1 * obs_conf,
            )
            ctx = self._edge_context_stability(t.src, t.dst)
            e.context_stability = max(
                0.0,
                min(1.0, 0.85 * float(getattr(e, "context_stability", 0.0) or 0.0) + 0.15 * ctx),
            )

        act_sorted = sorted(pattern, key=lambda x: x[1], reverse=True)[:8]
        ids = [k for k, _ in act_sorted]
        for a in ids:
            for b in ids:
                if a == b:
                    continue
                for c in ids:
                    if c in (a, b):
                        continue
                    if self._has_edge(a, b) and self._has_edge(b, c) and not self._has_edge(a, c):
                        te = self._get_or_create_episodic_edge(
                            a,
                            c,
                            "assoziation",
                            w_init=self.triangle_create_w,
                        )
                        te.evidence = max(float(getattr(te, "evidence", 0.0) or 0.0), 0.12)
                        te.confidence = max(float(getattr(te, "confidence", 0.0) or 0.0), 0.18)
                        te.context_stability = max(
                            float(getattr(te, "context_stability", 0.0) or 0.0),
                            self._edge_context_stability(a, c) * 0.75,
                        )

        for src, edges in list(self.episodic_edges.items()):
            kept = []
            for e in edges:
                e.gewicht = max(0.01, min(0.99, e.gewicht * self.episodic_weight_decay))
                e.evidence = max(0.0, float(getattr(e, "evidence", 0.0) or 0.0) * 0.994)
                e.confidence = max(
                    0.0,
                    min(1.0, float(getattr(e, "confidence", 0.0) or 0.0) * 0.998),
                )
                e.context_stability = max(
                    0.0,
                    min(1.0, float(getattr(e, "context_stability", 0.0) or 0.0) * 0.998),
                )
                if e.gewicht >= 0.05 or e.evidence >= 0.28:
                    kept.append(e)
            self.episodic_edges[src] = kept

        self.konsolidiere_episodisch()
        self._learn_ensemble_coactivation(pattern)

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

            # Hippocampus-Statistiken: Evidenz + Konfidenz + Kontextstabilität.
            e.evidence = max(0.0, float(getattr(e, "evidence", 0.0) or 0.0))
            e.evidence = min(999.0, e.evidence + 0.35 + float(max(0.0, t.contrib)))

            obs_conf = max(0.0, min(1.0, float(t.contrib) * 1.1))
            e.confidence = max(
                obs_conf,
                0.86 * float(getattr(e, "confidence", 0.0) or 0.0) + 0.14 * obs_conf,
            )

            ctx = self._edge_context_stability(t.src, t.dst)
            e.context_stability = max(
                0.0,
                min(
                    1.0,
                    0.82 * float(getattr(e, "context_stability", 0.0) or 0.0) + 0.18 * ctx,
                ),
            )

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
                        te = self._get_or_create_episodic_edge(
                            a,
                            c,
                            "assoziation",
                            w_init=self.triangle_create_w,
                        )
                        te.evidence = max(float(getattr(te, "evidence", 0.0) or 0.0), 0.15)
                        te.confidence = max(float(getattr(te, "confidence", 0.0) or 0.0), 0.2)
                        te.context_stability = max(
                            float(getattr(te, "context_stability", 0.0) or 0.0),
                            self._edge_context_stability(a, c) * 0.8,
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
                e.evidence = max(0.0, float(getattr(e, "evidence", 0.0) or 0.0) * 0.992)
                e.confidence = max(
                    0.0,
                    min(1.0, float(getattr(e, "confidence", 0.0) or 0.0) * 0.997),
                )
                e.context_stability = max(
                    0.0,
                    min(1.0, float(getattr(e, "context_stability", 0.0) or 0.0) * 0.998),
                )
                if e.gewicht >= 0.06 or e.evidence >= 0.35:
                    kept.append(e)
            self.episodic_edges[src] = kept

        self.konsolidiere_episodisch()
        self._learn_ensemble_coactivation(pattern)

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
        e = Verbindung(
            ziel=dst,
            gewicht=max(0.01, min(w_init, 0.99)),
            typ=typ,
            evidence=0.0,
            confidence=0.0,
            context_stability=0.0,
        )
        self.konzepte[src].verbindungen.append(e)
        return e

    def konsolidiere_episodisch(self):
        for src, edges in self.episodic_edges.items():
            for e in edges:
                score = self._consolidation_score(e)
                conf = float(getattr(e, "confidence", 0.0) or 0.0)
                if (
                    e.gewicht < self.consolidate_threshold
                    and score < self.consolidate_score_threshold
                ):
                    continue
                if (
                    conf < self.consolidate_min_confidence
                    and score < (self.consolidate_score_threshold + 0.08)
                ):
                    continue

                transfer = min(1.0, max(0.0, score))
                w_init = e.gewicht * self.consolidate_ratio * (0.75 + 0.45 * transfer)
                se = self._get_or_create_semantic_edge(
                    src,
                    e.ziel,
                    self._canon_type(e.typ),
                    w_init=w_init,
                )
                se.gewicht = min(0.99, max(se.gewicht, w_init))
                se.gewicht = min(0.99, se.gewicht + self.consolidate_boost * (0.8 + 0.4 * transfer))
                se.evidence = min(
                    999.0,
                    float(getattr(se, "evidence", 0.0) or 0.0) + 0.5 * transfer,
                )
                se.confidence = max(
                    float(getattr(se, "confidence", 0.0) or 0.0),
                    min(1.0, 0.6 * transfer + 0.4 * conf),
                )
                se.context_stability = max(
                    float(getattr(se, "context_stability", 0.0) or 0.0),
                    float(getattr(e, "context_stability", 0.0) or 0.0),
                )

    # -------------------------
    # Speichern (JSONL empfohlen)
    # -------------------------

    def speichere_model(self, datei: str, episodic_datei: Optional[str] = None):
        if self.cluster_mode_enabled:
            self._rebuild_cluster_index()
            self._rebuild_cluster_bridges()
        self._rebuild_ensembles()
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
                    "cluster": (self.konzepte[cid].cluster_id or ""),
                    "contexts": sorted(set(self.konzepte[cid].context_ids or [])),
                    "ensembles": sorted(set(self.konzepte[cid].ensemble_ids or [])),
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
                        "evidence": round(float(getattr(v, "evidence", 0.0) or 0.0), 4),
                        "confidence": round(float(getattr(v, "confidence", 0.0) or 0.0), 4),
                        "context": round(float(getattr(v, "context_stability", 0.0) or 0.0), 4),
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
                        "cluster": (self.konzepte[cid].cluster_id or ""),
                        "contexts": sorted(set(self.konzepte[cid].context_ids or [])),
                        "ensembles": sorted(set(self.konzepte[cid].ensemble_ids or [])),
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
                            "evidence": round(float(getattr(v, "evidence", 0.0) or 0.0), 4),
                            "confidence": round(float(getattr(v, "confidence", 0.0) or 0.0), 4),
                            "context": round(float(getattr(v, "context_stability", 0.0) or 0.0), 4),
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
