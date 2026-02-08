# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import subprocess
from typing import Dict, List, Optional, Tuple

from llm_core.types import Konzept, TraceItem


class BrocaMixin:
    # -------------------------
    # Init
    # -------------------------

    def _init_broca(self, lm_cmd: Optional[str], lm_callable):
        self.lm_cmd = lm_cmd or os.environ.get("LLM_CMD")
        self.lm_callable = lm_callable
        self.lm_timeout = 12
        self.use_lm_default = True
        self.explain_output = True
        self.broca_max_sents = 3
        # Fokus-Gewichtung (nahe an der Frage bleiben)
        self.focus_boost_src = 1.7
        self.focus_boost_dst = 1.3
        self.focus_penalty_other = 0.85

    # -------------------------
    # Output
    # -------------------------

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

    def versprachliche(
        self,
        denkmuster: List[Tuple[str, float]],
        intent: str = "OTHER",
        trace: Optional[List[TraceItem]] = None,
        use_lm: bool = True,
        focus_ids: Optional[List[str]] = None,
    ) -> str:
        if not denkmuster:
            return "Ich weiß das nicht."
        focus = focus_ids or [k for k, _ in denkmuster[:2]]
        context = [k for k, _ in denkmuster[2:8]]

        if use_lm:
            lm_text = self._lm_generate(denkmuster, intent=intent, trace=trace, focus_ids=focus)
            if lm_text:
                if self.explain_output and trace:
                    t0 = trace[0]
                    lm_text = lm_text.rstrip() + f" (Trace: {t0.src} → {t0.dst} / {t0.typ})"
                return lm_text

        sentences: List[str] = []
        seen = set()
        focus_edges = self._rank_focus_edges(focus, intent=intent)
        ranked = focus_edges if focus_edges else self._rank_candidate_edges(denkmuster, intent=intent, focus_ids=focus)
        for rel in ranked:
            if len(sentences) >= self.broca_max_sents:
                break
            tpl = self._template_for_type(rel["typ"])
            s = tpl.format(self._label_for_output(rel["src"]), self._label_for_output(rel["dst"]))
            key = s.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            sentences.append(s)

        if not sentences and denkmuster:
            sentences.append(f"Das zentrale Konzept ist {self._label_for_output(denkmuster[0][0])}.")

        extra = []
        focus_set = set(focus)
        if focus_set:
            for k in context:
                if k in focus_set:
                    continue
                # only allow extras that are directly linked to focus concepts
                linked = False
                for f in focus_set:
                    if self._has_edge_any(f, k) or self._has_edge_any(k, f):
                        linked = True
                        break
                if linked:
                    extra.append(self._label_for_output(k))
                if len(extra) >= 3:
                    break
        else:
            extra = [self._label_for_output(k) for k in context][:3]
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

    def _rank_candidate_edges(
        self,
        denkmuster: List[Tuple[str, float]],
        intent: str,
        focus_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, object]]:
        aktive = {k for k, _ in denkmuster}
        focus_set = set(focus_ids or [])
        out: List[Dict[str, object]] = []
        focus_edges: List[Dict[str, object]] = []
        for src in aktive:
            for e in self.konzepte.get(src, Konzept(src)).verbindungen:
                if e.ziel in aktive:
                    score = e.gewicht * self._gate(e.typ, intent)
                    if focus_set:
                        if src in focus_set:
                            score *= self.focus_boost_src
                        elif e.ziel in focus_set:
                            score *= self.focus_boost_dst
                        else:
                            score *= self.focus_penalty_other
                    item = {"score": score, "src": src, "dst": e.ziel, "typ": e.typ, "layer": "semantic"}
                    out.append(item)
                    if focus_set and (src in focus_set or e.ziel in focus_set):
                        focus_edges.append(item)
            for e in self.episodic_edges.get(src, []):
                if e.ziel in aktive:
                    score = (e.gewicht * 0.9) * self._gate(e.typ, intent)
                    if focus_set:
                        if src in focus_set:
                            score *= self.focus_boost_src
                        elif e.ziel in focus_set:
                            score *= self.focus_boost_dst
                        else:
                            score *= self.focus_penalty_other
                    item = {"score": score, "src": src, "dst": e.ziel, "typ": e.typ, "layer": "episodic"}
                    out.append(item)
                    if focus_set and (src in focus_set or e.ziel in focus_set):
                        focus_edges.append(item)
        if focus_edges:
            focus_edges.sort(key=lambda x: x["score"], reverse=True)
            return focus_edges
        out.sort(key=lambda x: x["score"], reverse=True)
        return out

    def _rank_focus_edges(self, focus_ids: List[str], intent: str) -> List[Dict[str, object]]:
        if not focus_ids:
            return []
        out: List[Dict[str, object]] = []
        for src in focus_ids:
            if src in self.konzepte:
                for e in self.konzepte[src].verbindungen:
                    score = e.gewicht * self._gate(e.typ, intent)
                    out.append({"score": score, "src": src, "dst": e.ziel, "typ": e.typ, "layer": "semantic"})
            for e in self.episodic_edges.get(src, []):
                score = (e.gewicht * 0.9) * self._gate(e.typ, intent)
                out.append({"score": score, "src": src, "dst": e.ziel, "typ": e.typ, "layer": "episodic"})
        out.sort(key=lambda x: x["score"], reverse=True)
        return out

    def _has_edge_any(self, src: str, dst: str) -> bool:
        if src in self.konzepte:
            for e in self.konzepte[src].verbindungen:
                if e.ziel == dst:
                    return True
        if src in self.episodic_edges:
            for e in self.episodic_edges[src]:
                if e.ziel == dst:
                    return True
        return False

    def _lm_generate(
        self,
        denkmuster: List[Tuple[str, float]],
        intent: str,
        trace: Optional[List[TraceItem]],
        focus_ids: Optional[List[str]] = None,
    ) -> Optional[str]:
        if not (self.lm_callable or self.lm_cmd):
            return None
        focus_ids = focus_ids or [k for k, _ in denkmuster[:2]]
        aktive = [self._label_for_output(k) for k, _ in denkmuster[:8]]
        rels = self._rank_focus_edges(focus_ids, intent=intent)
        if not rels:
            rels = self._rank_candidate_edges(denkmuster, intent=intent, focus_ids=focus_ids)
        rels = rels[:5]
        rel_lines = [
            f"{self._label_for_output(r['src'])} -{r['typ']}-> {self._label_for_output(r['dst'])} (w={r['score']:.2f}, {r['layer']})"
            for r in rels
        ]
        prompt = (
            "Du bist Broca und formulierst kurze, natürliche deutsche Sätze.\n"
            "Nutze die folgenden Konzepte und Relationen, erfinde keine neuen Fakten.\n"
            "Gib 1 bis 3 Sätze aus, keine Listen.\n\n"
            f"Intent: {intent}\n"
            f"Fokus: {', '.join([self._label_for_output(k) for k in focus_ids])}\n"
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
