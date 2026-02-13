# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
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
        self.cause_boost = 1.35
        self.cause_types = {
            "verursacht", "durch", "streut", "verstärkt", "verstaerkt",
            "ermöglicht", "ermoeglicht", "notwendig_für", "notwendig_für", "notwendig_fuer",
            "benötigt", "benötigt", "benoetigt", "braucht",
            "verursacht_durch", "verursacht_von",
        }

    # -------------------------
    # Output
    # -------------------------

    def _label_for_output(self, kid: str) -> str:
        k = self.konzepte.get(kid)
        if not k or not k.labels:
            return kid
        labels = [lab for lab in k.labels if lab]
        if not labels:
            return kid
        def score(lab: str) -> Tuple[int, int]:
            s = 0
            if lab == kid:
                s += 3
            if lab[:1].isupper():
                s += 2
            if " " in lab:
                s += 1
            if lab.islower():
                s -= 1
            return (s, len(lab))

        labels.sort(key=score, reverse=True)
        return labels[0]

    def _is_junk_node(self, kid: str) -> bool:
        if not kid:
            return True
        if len(kid) <= 2 and not kid.isupper():
            return True
        low = kid.lower()
        junk_ids = {
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
        }
        if low in junk_ids:
            return True
        tokens = [t for t in kid.split("_") if t]
        if not tokens:
            return True
        junk_tokens = {
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
        }
        junk_count = 0
        for t in tokens:
            t_low = t.lower()
            if len(t_low) <= 2 or t_low in junk_tokens:
                junk_count += 1
        return (junk_count / len(tokens)) >= 0.6

    def versprachliche(
        self,
        denkmuster: List[Tuple[str, float]],
        intent: str = "OTHER",
        trace: Optional[List[TraceItem]] = None,
        use_lm: bool = True,
        focus_ids: Optional[List[str]] = None,
        memory_hits: Optional[List[Dict[str, object]]] = None,
    ) -> str:
        if not denkmuster:
            return "Ich weiß das nicht."
        focus = focus_ids or [k for k, _ in denkmuster[:2]]
        context = [k for k, _ in denkmuster[2:8]]
        focus_edges_for_intent = self._rank_focus_edges(focus, intent=intent)
        primary_memory = self._best_memory_text(memory_hits or [], focus)
        intent_memory = self._best_intent_sentence(memory_hits or [], focus, intent)
        wiki_snip = self._wiki_snippet_for_focus(focus, intent=intent)
        if intent_memory:
            primary_memory = intent_memory
        elif wiki_snip and (not primary_memory or len(primary_memory) < 60):
            primary_memory = wiki_snip
        if primary_memory:
            if intent in {"CAUSE", "HOW", "WHERE", "PARTS", "PROPS", "DEF"}:
                if intent == "PROPS" and focus_edges_for_intent:
                    primary_memory = ""
                else:
                    return primary_memory

        if use_lm:
            lm_text = self._lm_generate(
                denkmuster,
                intent=intent,
                trace=trace,
                focus_ids=focus,
                memory_hits=memory_hits,
            )
            if lm_text:
                if self.explain_output and trace:
                    t0 = trace[0]
                    lm_text = lm_text.rstrip() + f" (Trace: {t0.src} → {t0.dst} / {t0.typ})"
                return lm_text

        sentences: List[str] = []
        seen = set()
        used_ids = set()
        focus_edges = focus_edges_for_intent
        if focus_edges:
            ranked = focus_edges
        else:
            ranked = self._rank_candidate_edges(
                denkmuster,
                intent=intent,
                focus_ids=focus,
            )
        for rel in ranked:
            if len(sentences) >= self.broca_max_sents:
                break
            tpl = self._template_for_type(rel["typ"])
            src_label = self._label_for_output(rel["src"])
            dst_label = self._label_for_output(rel["dst"])
            s = tpl.format(src_label, dst_label)
            sym = self._is_symmetric_type(rel["typ"], tpl)
            if sym:
                key = "sym:" + "|".join(sorted([src_label.lower(), dst_label.lower()]))
            else:
                key = f"dir:{src_label.lower()}->{dst_label.lower()}:{rel['typ']}"
            if key in seen:
                continue
            seen.add(key)
            sentences.append(s)
            used_ids.add(rel["src"])
            used_ids.add(rel["dst"])

        if not sentences and denkmuster:
            sentences.append(
                f"Das zentrale Konzept ist {self._label_for_output(denkmuster[0][0])}."
            )

        extra = []
        focus_set = set(focus)
        if intent in {"CAUSE", "PROPS", "COMPARE"}:
            extra = []
        elif focus_set:
            for k in context:
                if k in focus_set:
                    continue
                if self._is_junk_node(k):
                    continue
                if k in used_ids:
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

        # Optional: memory snippet for freieres Denken (ohne LM)
        if memory_hits:
            snippet = self._memory_snippet(memory_hits, focus)
            if snippet:
                sentences.append(snippet)

        s1 = " ".join(sentences)
        if self.explain_output and trace:
            t0 = self._select_trace(trace, focus)
            if t0.src and t0.dst:
                s1 = s1.rstrip() + f" (Trace: {t0.src} → {t0.dst} / {t0.typ})"
        return s1

    def _memory_snippet(self, memory_hits: List[Dict[str, object]], focus: List[str]) -> str:
        if not memory_hits:
            return ""
        text = self._best_memory_text(memory_hits, focus)
        if not text:
            return ""
        if len(text) > 160:
            text = text[:157].rstrip() + "..."
        return "Erinnerung: " + text

    def _clean_snippet_text(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""
        # fix common OCR artifacts seen in imports
        t = t.replace("Verb und", "Verbund")
        t = t.replace("Grundlagenft", "Grundlagen- und")
        t = re.sub(r"Grundlagen- und\s+und", "Grundlagen- und", t)
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    def _is_bad_sentence(self, text: str) -> bool:
        s = (text or "").strip()
        if not s:
            return True
        low = s.lower()
        if " ist i." in low or low.endswith(" i.") or low.endswith(" i"):
            return True
        if s[:1].islower():
            return True
        if len(s) < 20:
            return True
        return False

    def _select_snippet(self, text: str) -> str:
        clean = self._clean_snippet_text(text)
        if not clean:
            return ""
        parts = re.split(r"(?<=[.!?])\s+", clean)
        good = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if self._is_bad_sentence(p):
                continue
            good.append(p)
            if len(good) >= 2:
                break
        if good:
            return " ".join(good).strip()
        return clean

    def _intent_keywords(self, intent: str) -> List[str]:
        if intent == "PARTS":
            return ["besteht aus", "setzt sich zusammen", "umfasst"]
        if intent == "CAUSE":
            return ["verursacht", "bedingt", "entsteht durch", "führt zu"]
        if intent == "WHERE":
            return ["lebt in", "kommt in", "ist in", "vorkommt in"]
        if intent == "PROPS":
            return ["eigenschaft", "zeichnet sich", "hat", "weist", "charakteristisch"]
        if intent == "COMPARE":
            return ["unterschied", "im unterschied", "im gegensatz", "während"]
        return []

    def _select_sentence_by_keywords(self, text: str, keywords: List[str]) -> str:
        if not text or not keywords:
            return ""
        clean = self._clean_snippet_text(text)
        if not clean:
            return ""
        parts = re.split(r"(?<=[.!?])\s+", clean)
        for p in parts:
            s = p.strip()
            if not s:
                continue
            low = s.lower()
            if any(k in low for k in keywords) and not self._is_bad_sentence(s):
                return s
        return ""

    def _best_intent_sentence(
        self,
        memory_hits: List[Dict[str, object]],
        focus: List[str],
        intent: str,
    ) -> str:
        if not memory_hits:
            return ""
        keywords = self._intent_keywords(intent)
        if not keywords:
            return ""
        focus_labels = set()
        primary_labels = set()
        for f in focus or []:
            lab = self._label_for_output(f).lower()
            focus_labels.add(lab)
            focus_labels.add(f.lower())
        if focus:
            f0 = focus[0]
            primary_labels.add(self._label_for_output(f0).lower())
            primary_labels.add(f0.lower())

        for hit in memory_hits:
            score = float(hit.get("score") or 0.0)
            if score < 0.2:
                continue
            t = str(hit.get("text") or "").strip()
            if not t:
                continue
            tl = t.lower()
            if "features:" in tl:
                continue
            if focus_labels and not any(fl in tl for fl in focus_labels):
                continue
            if primary_labels and not any(pl in tl for pl in primary_labels):
                continue
            sent = self._select_sentence_by_keywords(t, keywords)
            if sent:
                return sent
        return ""

    def _best_memory_text(
        self,
        memory_hits: List[Dict[str, object]],
        focus: List[str],
    ) -> str:
        if not memory_hits:
            return ""
        focus_labels = set()
        primary_labels = set()
        for f in focus or []:
            lab = self._label_for_output(f).lower()
            focus_labels.add(lab)
            focus_labels.add(f.lower())
        if focus:
            f0 = focus[0]
            primary_labels.add(self._label_for_output(f0).lower())
            primary_labels.add(f0.lower())
        preferred = []
        secondary = []
        fallback = []
        for hit in memory_hits:
            score = float(hit.get("score") or 0.0)
            if score < 0.25:
                continue
            meta = hit.get("meta") or {}
            t = str(hit.get("text") or "").strip()
            if not t:
                continue
            tl = t.lower()
            if "features:" in tl:
                continue
            if focus_labels and not any(fl in tl for fl in focus_labels):
                continue
            is_primary = bool(primary_labels) and any(pl in tl for pl in primary_labels)
            if meta.get("type") in {"node", "edge"}:
                fallback.append(t)
            elif is_primary:
                preferred.append(t)
            else:
                secondary.append(t)
        for bucket in (preferred, secondary, fallback):
            for t in bucket:
                if len(t) >= 40:
                    clean = t.replace("\n", " ").strip()
                    short = self._select_snippet(clean)
                    return short if short else clean
        return ""

    def _wiki_file_map(self) -> Dict[str, Path]:
        if hasattr(self, "_wiki_cache") and isinstance(self._wiki_cache, dict):
            return self._wiki_cache
        wiki_dir = Path(getattr(self, "base_dir", ".")) / "data_import" / "wikipedia"
        mapping: Dict[str, Path] = {}
        if wiki_dir.exists():
            for p in wiki_dir.glob("*.txt"):
                stem = p.stem
                stem = re.sub(r"^\d+_", "", stem)
                base = stem.lower()
                mapping[base] = p
                # simple adjective ending variants (menschlicher -> menschliche)
                parts = base.split("_")
                for i, part in enumerate(parts):
                    if part.endswith("er") and len(part) > 3:
                        alt = part[:-1]
                        v = parts[:]
                        v[i] = alt
                        mapping.setdefault("_".join(v), p)
        self._wiki_cache = mapping
        return mapping

    def _slugify_label(self, text: str) -> str:
        t = (text or "").strip().lower()
        if not t:
            return ""
        t = t.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
        t = re.sub(r"[^a-z0-9]+", "_", t)
        t = re.sub(r"_+", "_", t).strip("_")
        return t

    def _wiki_snippet_for_focus(self, focus: List[str], intent: Optional[str] = None) -> str:
        if not focus:
            return ""
        mapping = self._wiki_file_map()
        for kid in focus:
            k = self.konzepte.get(kid)
            candidates = [kid] + (k.labels or [])
            for cand in candidates:
                slug = self._slugify_label(cand)
                if not slug:
                    continue
                path = mapping.get(slug)
                if not path:
                    continue
                try:
                    raw = path.read_text(encoding="utf-8")
                except Exception:
                    continue
                text = raw.replace("\r", " ").replace("\n", " ").strip()
                if intent:
                    sent = self._select_sentence_by_keywords(text, self._intent_keywords(intent))
                    if sent:
                        return sent
                short = self._select_snippet(text)
                return short if short else text
        return ""

    def _select_trace(self, trace: List[TraceItem], focus: List[str]) -> TraceItem:
        if not trace:
            return TraceItem(tick=0, src="", dst="", typ="", contrib=0.0, layer="")
        fset = set(focus or [])
        if fset:
            for t in trace:
                if self._is_junk_node(t.src) or self._is_junk_node(t.dst):
                    continue
                if t.src in fset or t.dst in fset:
                    return t
        for t in trace:
            if self._is_junk_node(t.src) or self._is_junk_node(t.dst):
                continue
            return t
        return TraceItem(tick=0, src="", dst="", typ="", contrib=0.0, layer="")

    def _is_symmetric_type(self, typ: str, template: str) -> bool:
        # Treat unknown templates as symmetric to avoid reversed duplicates
        symmetric = {"assoziation", "gelernt"}
        if typ in symmetric:
            return True
        if template == "{} und {} sind eng miteinander verbunden.":
            return True
        return False

    def _is_cause_type(self, typ: str) -> bool:
        t = self._nfc(typ) if hasattr(self, "_nfc") else typ
        return t in self.cause_types

    def _template_for_type(self, typ: str) -> str:
        t = self._nfc(typ) if hasattr(self, "_nfc") else typ
        templates = {
            "teil_von": "{} ist ein wesentlicher Teil von {}.",
            "hat": "{} hat oder besitzt {}.",
            "ist": "{} ist im Grunde {}.",
            "eigenschaft": "{} hat die charakteristische Eigenschaft {}.",
            "eigenschaft_von": "{} ist eine Eigenschaft von {}.",
            "prozess": "{} ist ein Prozess, in dem {} zentral ist.",
            "ermöglicht": "{} ermöglicht {}.",
            "ermoeglicht": "{} ermöglicht {}.",
            "besteht_aus": "{} setzt sich zusammen aus {}.",
            "benötigt": "{} benötigt {} als Voraussetzung.",
            "benötigt": "{} benötigt {} als Voraussetzung.",
            "benoetigt": "{} benötigt {} als Voraussetzung.",
            "braucht": "{} braucht {} zum Funktionieren.",
            "verursacht": "{} verursacht {}.",
            "notwendig_für": "{} ist notwendig für {}.",
            "notwendig_für": "{} ist notwendig für {}.",
            "notwendig_fuer": "{} ist notwendig für {}.",
            "gehört_zu": "{} gehört zu {}.",
            "gehört_zu": "{} gehört zu {}.",
            "gehoert_zu": "{} gehört zu {}.",
            "lebt_in": "{} lebt in {}.",
            "gelernt": "{} und {} stehen in enger Beziehung.",
            "assoziation": "{} steht in Zusammenhang mit {}.",
            "farbe": "{} hat die Farbe {}.",
            "enthält": "{} enthält {}.",
            "enthält": "{} enthält {}.",
            "enthaelt": "{} enthält {}.",
            "durch": "{} ist durch {} geprägt.",
            "streut": "{} streut {}.",
            "verstärkt": "{} verstärkt {}.",
            "verstaerkt": "{} verstärkt {}.",
            "sichtbar_in": "{} ist sichtbar in {}.",
            "sichtbar_für": "{} ist sichtbar für {}.",
            "sichtbar_für": "{} ist sichtbar für {}.",
            "sichtbar_fu": "{} ist sichtbar für {}.",
            "durchlässig_für": "{} ist durchlässig für {}.",
            "durchlässig_für": "{} ist durchlässig für {}.",
            "durchlaessig_fuer": "{} ist durchlässig für {}.",
            "kann_erleiden": "{} kann {} erleiden.",
            "verursacht_durch": "{} wird durch {} verursacht.",
            "verursacht_von": "{} wird durch {} verursacht.",
            "zeigt": "{} zeigt {}.",
            "filtert": "{} filtert {}.",
            "ist_typ": "{} ist ein Typ von {}.",
            "in": "{} ist in {}.",
            "von": "{} stammt von {}.",
            "absorbiert": "{} absorbiert {}.",
            "wahrgenommen_als": "{} wird als {} wahrgenommen.",
            "wahrgenommen_durch": "{} wird durch {} wahrgenommen.",
            "bestimmt": "{} bestimmt {}.",
            "bestimmt_durch": "{} wird durch {} bestimmt.",
            "beeinflusst": "{} beeinflusst {}.",
        }
        return templates.get(t, "{} und {} sind eng miteinander verbunden.")

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
            if self._is_junk_node(src):
                continue
            for e in self.konzepte.get(src, Konzept(src)).verbindungen:
                if e.ziel in aktive:
                    if self._is_junk_node(e.ziel):
                        continue
                    score = e.gewicht * self._gate(e.typ, intent)
                    if intent == "CAUSE" and self._is_cause_type(e.typ):
                        score *= self.cause_boost
                    if focus_set:
                        if src in focus_set:
                            score *= self.focus_boost_src
                        elif e.ziel in focus_set:
                            score *= self.focus_boost_dst
                        else:
                            score *= self.focus_penalty_other
                    item = {
                        "score": score,
                        "src": src,
                        "dst": e.ziel,
                        "typ": e.typ,
                        "layer": "semantic",
                    }
                    out.append(item)
                    if focus_set and (src in focus_set or e.ziel in focus_set):
                        focus_edges.append(item)
            for e in self.episodic_edges.get(src, []):
                if e.ziel in aktive:
                    if self._is_junk_node(e.ziel):
                        continue
                    score = (e.gewicht * 0.9) * self._gate(e.typ, intent)
                    if intent == "CAUSE" and self._is_cause_type(e.typ):
                        score *= self.cause_boost
                    if focus_set:
                        if src in focus_set:
                            score *= self.focus_boost_src
                        elif e.ziel in focus_set:
                            score *= self.focus_boost_dst
                        else:
                            score *= self.focus_penalty_other
                    item = {
                        "score": score,
                        "src": src,
                        "dst": e.ziel,
                        "typ": e.typ,
                        "layer": "episodic",
                    }
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
            if self._is_junk_node(src):
                continue
            if src in self.konzepte:
                for e in self.konzepte[src].verbindungen:
                    if self._is_junk_node(e.ziel):
                        continue
                    score = e.gewicht * self._gate(e.typ, intent)
                    if intent == "CAUSE" and self._is_cause_type(e.typ):
                        score *= self.cause_boost
                    out.append(
                        {
                            "score": score,
                            "src": src,
                            "dst": e.ziel,
                            "typ": e.typ,
                            "layer": "semantic",
                        }
                    )
            for e in self.episodic_edges.get(src, []):
                if self._is_junk_node(e.ziel):
                    continue
                score = (e.gewicht * 0.9) * self._gate(e.typ, intent)
                if intent == "CAUSE" and self._is_cause_type(e.typ):
                    score *= self.cause_boost
                out.append(
                    {
                        "score": score,
                        "src": src,
                        "dst": e.ziel,
                        "typ": e.typ,
                        "layer": "episodic",
                    }
                )
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
        memory_hits: Optional[List[Dict[str, object]]] = None,
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
            (
                f"{self._label_for_output(r['src'])} -{r['typ']}-> "
                f"{self._label_for_output(r['dst'])} "
                f"(w={r['score']:.2f}, {r['layer']})"
            )
            for r in rels
        ]
        mem_lines = []
        for hit in (memory_hits or [])[:3]:
            txt = str(hit.get("text") or "").strip()
            if txt:
                if len(txt) > 140:
                    txt = txt[:137].rstrip() + "..."
                mem_lines.append(f"- {txt}")
        prompt = (
            "Du bist Broca und formulierst kurze, natürliche deutsche Sätze.\n"
            "Nutze die folgenden Konzepte und Relationen, erfinde keine neuen Fakten.\n"
            "Gib 1 bis 3 Sätze aus, keine Listen.\n\n"
            f"Intent: {intent}\n"
            f"Fokus: {', '.join([self._label_for_output(k) for k in focus_ids])}\n"
            f"Konzepte: {', '.join(aktive)}\n"
            "Relationen:\n" + "\n".join(rel_lines) + "\n"
            + ("Erinnerungen:\n" + "\n".join(mem_lines) + "\n" if mem_lines else "")
        )

        if self.lm_callable:
            try:
                txt = self.lm_callable(
                    {
                        "prompt": prompt,
                        "intent": intent,
                        "concepts": aktive,
                        "relations": rels,
                        "trace": trace,
                    }
                )
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
