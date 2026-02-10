# -*- coding: utf-8 -*-
# ============================================================
# Attention-Cluster LLM – Data Import (Wikipedia + Tatoeba)
# Datei: llm_training/import/import_training_data.py
#
# VERSION: 1.2.6
# STATUS: STABIL
#
# ROLLE:
# - Lädt Knowledge (Wikipedia) und Form (Tatoeba) als Trainings-.txt
# - Normalisiert Import-Texte, damit kein „Fragment-Vokab“ entsteht
#
# PIPELINE:
# - Zuerst FORM (Tatoeba), dann KNOWLEDGE (Wikipedia)
# - Import-Zielordner: llm_training/data/**/01_wikipedia-import
#
# PARETO-FIX (WICHTIG):
# - Repariert Kürzungs-Bindestriche in Aufzählungen:
#   "Knochen-, Haut- und Gewebeabdrücken"
#   "Knochen, Haut- und Gewebeabdrücken"
#   -> "Knochenabdrücken, Hautabdrücken und Gewebeabdrücken"
#
# - Entfernt Soft-Hyphen / Zero-Width Artefakte
# - Vereinheitlicht force_update (nur 1 Stelle)
#
# ÄNDERUNGEN:
# 1.2.6 – FIX: „o. g.“ (oben genannt) robust reparieren:
#         PRE: "o.\n g." -> "o. g."  | POST: "... o." + "g. ..." -> "... o. g. ...".
#         FIX: Bindestrich-Schutz bei Abkürzungs-Komposita (Hyphen MUSS bleiben):
#              - PRE: "KI- Verordnung" -> "KI-Verordnung"
#              - PRE: "KIVerordnung"  -> "KI-Verordnung" (konservativ über Whitelist)
#              - PRE: "NonKISystem"   -> "Non-KI-System"
#              - Strip-Fix: "KI-" / "EU-" / "IT-" / "LLM-" werden NICHT mehr entfernt
# 1.2.5 – FIX: Robuste d. h. / z. b. Reparatur auch bei Satzsplit:
#              "... d." + "h. ..."  -> "... d. h. ..."
#              "... d." + "auf ..." -> "... d. h. auf ..."
#              "... z." + "b. ..."  -> "... z. b. ..."
#              "... z." + "von ..." -> "... z. b. von ..."
# 1.2.4 – FIX: Satzsplit-Repair für Abkürzungs-Fragmente:
#         "..., d." + "auf ..." -> "..., d. h. auf ...".
# 1.2.3 – FIX: FORM wird vor KNOWLEDGE importiert.
#         FIX: Wiki-Listen/Bullets werden vor Satzsplit in saubere Satzgrenzen überführt.
#         FIX: Repariert häufige Abkürzungen über Zeilenumbrüche
#              (z. B. "d.\n h." -> "d. h.", "z.\n b." -> "z. b.").
#         FIX: Rettet häufige Fragment-Fälle "d.\n auf" -> "d. h. auf"
#              und "z.\n von" -> "z. b. von".
#         ADD: sentence_limit als Alias für line_limit (tatsächlich: Satz-Limit nach Split).
#         ADD: Punctuation-Cleanup (". .", "..", ":.") um Split-Müll zu vermeiden.
# 1.2.2 – FIX: Enum-Expansion nur noch bei echten Dash-Fragmenten.
#         Verhindert False-Positives wie "Sprache oder Deutsch" -> "Sprachesch..."
#         und macht normalize() wieder idempotent.
# 1.2.1 – Fix: erkennt auch Fälle, in denen das Kopfwort nur im letzten Token steckt
#         (z.B. "... Knochen, Haut- und Gewebeabdrücken")
#         + besseres Compound-Splitting (Gewebe|abdrücken)
#         + optionaler Last-Resort-Strip für "wort-" (ohne sinnvolle Expansion)
#
# ============================================================

from __future__ import annotations

import json
import os
import re
import unicodedata
import urllib.parse
import urllib.request
from typing import List, Optional, Tuple

try:
    import requests  # type: ignore
except Exception:
    requests = None


# ============================================================
# Config
# ============================================================

DEFAULT_CONFIG = {
    "knowledge_topics": ["KI"],
    "form_sentence_count": 50,

    # IMPORTANT:
    # line_limit ist historisch benannt, tatsächlich ist es ein Satz-Limit
    # nach dem Satzsplit (fetch_wikipedia -> sentences[:limit]).
    "line_limit": 10,
    # Alias (optional). Wenn gesetzt, überschreibt es line_limit.
    "sentence_limit": None,

    "user_agent": "MobileBot/1.0",

    # ✅ nur 1 Flag
    "force_update": True,

    # Normalisierung toggles
    "normalize_import_text": True,
    "expand_dash_enumerations": True,
    "strip_invisible_unicode": True,

    # optional: extra safety-net gegen "wort-"
    "dash_fragment_safety_net": True,

    # Last-Resort: entfernt "wort-" -> "wort" (nur wenn Expansion nicht greift)
    "strip_dangling_hyphen_tokens": True,

    # Wiki-Listen + Abkürzungen reparieren
    "normalize_wiki_lists": True,
    "repair_common_abbrev": True,
    "protect_abbrev_for_split": True,
    "fix_camelcase_compounds": True,
    "fix_glued_conjunctions": True,
    "strip_parentheticals": True,

    # Debug
    "print_normalize_stats": True,
}


def _http_get_json(url: str, params: dict, headers: dict, timeout: int) -> dict:
    """
    Small HTTP helper that uses requests when available, otherwise falls back to urllib.
    Keeps the script runnable without external dependencies.
    """
    if requests is not None:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json() if r is not None else {}

    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{qs}", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        return json.loads(data.decode("utf-8"))


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f) or {}
            if isinstance(user_cfg, dict):
                cfg.update(user_cfg)
        except Exception:
            pass

    # defensiv: keys absichern
    cfg.setdefault("force_update", False)
    cfg.setdefault("normalize_import_text", True)
    cfg.setdefault("expand_dash_enumerations", True)
    cfg.setdefault("strip_invisible_unicode", True)
    cfg.setdefault("dash_fragment_safety_net", True)
    cfg.setdefault("strip_dangling_hyphen_tokens", True)
    cfg.setdefault("normalize_wiki_lists", True)
    cfg.setdefault("repair_common_abbrev", True)
    cfg.setdefault("protect_abbrev_for_split", True)
    cfg.setdefault("fix_camelcase_compounds", True)
    cfg.setdefault("fix_glued_conjunctions", True)
    cfg.setdefault("strip_parentheticals", True)
    cfg.setdefault("print_normalize_stats", True)

    # sentence_limit alias
    if cfg.get("sentence_limit") is None:
        cfg["sentence_limit"] = cfg.get("line_limit", 10)

    # force_update soll bool sein
    cfg["force_update"] = bool(cfg.get("force_update", False))
    return cfg


# ============================================================
# Helpers: Dateiname / Pfade
# ============================================================

def clean_filename(text: str) -> str:
    text = str(text or "")
    replacements = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", " ": "_"}
    for char, rep in replacements.items():
        text = text.replace(char, rep)
    return re.sub(r"[^a-zA-Z0-9_]", "", text).lower().strip("_") or "topic"


def setup_paths() -> dict:
    # Script liegt in data_import; Pfade relativ dazu auflösen
    this_dir = os.path.dirname(os.path.abspath(__file__))
    base = os.path.abspath(os.path.join(this_dir, ".."))

    paths = {
        "knowledge": os.path.join(base, "data_import", "wikipedia"),
        "form": os.path.join(base, "data_import", "form"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


# ============================================================
# Text Normalization
# ============================================================

_INVISIBLE_CHARS = [
    "\u00ad",  # SOFT HYPHEN
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZWNJ
    "\u200d",  # ZWJ
    "\ufeff",  # BOM / ZERO WIDTH NO-BREAK SPACE
]

# Normalisiere verschiedene Striche auf ASCII-Hyphen, weil Wikipedia gern "–" (en dash) nutzt
# hyphen, non-breaking, figure, en, em, minus
_DASH_CHARS = [
    "\u2010",
    "\u2011",
    "\u2012",
    "\u2013",
    "\u2014",
    "\u2212",
]

# Letter-Set (inkl. Umlaute/ß) für simple Prüfungen
_LOWER_EXTRA = "äöüß"
_UPPER_EXTRA = "ÄÖÜ"

# Placeholder, um Abkürzungs-Punkte beim Satzsplit zu schützen
_SPLIT_DOT = "<DOT>"


class _NormStats:
    def __init__(self):
        self.strip_invisible = 0
        self.dash_unified = 0
        self.enum_expanded = 0
        self.safety_net = 0
        self.dangling_hyphen_stripped = 0
        self.abbrev_repaired = 0
        self.bullets_normalized = 0
        self.punct_cleaned = 0
        self.split_abbrev_joined = 0

        # 1.2.6
        self.hyphen_space_fixed = 0
        self.abbrev_hyphen_restored = 0
        self.nonki_restored = 0

        # Split/quality fixes
        self.split_protected = 0
        self.camel_hyphen_fixed = 0
        self.glued_fixed = 0
        self.parentheticals_stripped = 0


def _is_lower(ch: str) -> bool:
    return ch.islower() or ch in _LOWER_EXTRA


def _is_upper(ch: str) -> bool:
    return ch.isupper() or ch in _UPPER_EXTRA


def _strip_invisible(text: str, st: _NormStats) -> str:
    if not text:
        return ""
    before = text
    for ch in _INVISIBLE_CHARS:
        text = text.replace(ch, "")
    if text != before:
        st.strip_invisible += 1
    return text


def _unify_dashes(text: str, st: _NormStats) -> str:
    if not text:
        return ""
    before = text
    for ch in _DASH_CHARS:
        text = text.replace(ch, "-")
    if text != before:
        st.dash_unified += 1
    return text


def _repair_common_abbrev(text: str, st: _NormStats) -> str:
    """
    Repariert häufige Abkürzungen, die in Wiki-Extracts durch Zeilenumbrüche zerbrechen,
    z.B. "d.\n h." oder "z.\n b.".
    Zusätzlich: Rettet häufige Fragment-Fälle wie "d.\n auf" und "z.\n von".
    (1.2.6) zusätzlich: "o.\n g." -> "o. g."
    """
    if not text:
        return ""

    # vereinheitliche Newlines für die Regex
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    before = t

    patterns = [
        (re.compile(r"\bd\s*\.\s*(?:\n|\s)+h\s*\.", re.IGNORECASE), "d. h."),
        (re.compile(r"\bz\s*\.\s*(?:\n|\s)+b\s*\.", re.IGNORECASE), "z. b."),
        (re.compile(r"\bu\s*\.\s*(?:\n|\s)+a\s*\.", re.IGNORECASE), "u. a."),
        (re.compile(r"\bi\s*\.\s*(?:\n|\s)+d\s*\.\s*(?:\n|\s)+r\s*\.", re.IGNORECASE), "i. d. r."),
        (re.compile(r"\bo\s*\.\s*(?:\n|\s)+g\s*\.", re.IGNORECASE), "o. g."),

        # Fragment-Rettung (ohne "h.") – sehr häufig in Listen:
        (re.compile(r"\bd\s*\.\s*(?:\n|\s)+auf\b", re.IGNORECASE), "d. h. auf"),
        (re.compile(r"\bd\s*\.\s*(?:\n|\s)+zu\b", re.IGNORECASE), "d. h. zu"),
        (re.compile(r"\bd\s*\.\s*(?:\n|\s)+an\b", re.IGNORECASE), "d. h. an"),
        (re.compile(r"\bz\s*\.\s*(?:\n|\s)+von\b", re.IGNORECASE), "z. b. von"),
    ]

    for rx, repl in patterns:
        t2 = rx.sub(repl, t)
        if t2 != t:
            st.abbrev_repaired += 1
            t = t2

    return t if t != before else text


def _normalize_wiki_lists(text: str, st: _NormStats) -> str:
    """
    Wiki-Extract enthält teils Bullet-Listen (•, -, *) und Zeilenumbrüche.
    Ziel: Bullet-Items als klare Satzgrenzen markieren, bevor wir Newlines entfernen.
    """
    if not text:
        return ""

    t = text.replace("\r\n", "\n").replace("\r", "\n")
    before = t

    # Bullet-Zeilen zu Satzgrenzen machen: "\n  • foo" -> ". foo"
    t = re.sub(r"\n\s*[\u2022•\-*]+\s+", ". ", t)

    # Mehrere Leerzeilen -> Satztrenner
    t = re.sub(r"\n{2,}", ". ", t)

    # Restliche Newlines: wenn vor Newline ein Wortzeichen steht, erzwingen wir Satzgrenze
    t = re.sub(r"(?<=[\wÄÖÜäöüß])\n\s+", ". ", t)

    if t != before:
        st.bullets_normalized += 1

    return t


def _punctuation_cleanup(text: str, st: _NormStats) -> str:
    """
    Glättet häufige Punctuation-Artefakte aus Wiki-Extracts, damit Satzsplit stabil bleibt:
    - ".." -> "."
    - ":." -> ":"
    - " . " / " ." -> "."
    """
    if not text:
        return ""
    before = text

    t = text
    t = re.sub(r"\.\.+", ".", t)          # mehrfache Punkte
    t = re.sub(r":\s*\.", ":", t)         # ":." -> ":"
    t = re.sub(r"\s+\.", ".", t)          # " ." -> "."
    t = t.replace(" ,", ",")              # ". ," -> ","
    t = re.sub(r"\s{2,}", " ", t).strip()

    if t != before:
        st.punct_cleaned += 1
    return t


# ------------------------------------------------------------
# Split-Protection + CamelCase/Glue-Fixes
# ------------------------------------------------------------

_CAMEL_HYPHEN_RE = re.compile(r"\b([A-Za-zÄÖÜäöüß]{2,}[a-zäöüß])([A-ZÄÖÜ][a-zäöüß]{2,})\b")
_GLUED_CONJ_RE = re.compile(r"\b([A-ZÄÖÜ][a-zäöüß]{2,})(und|oder|bzw)(\s+)")
_GLUED_BIS_RE = re.compile(r"\b([A-ZÄÖÜ][a-zäöüß]{2,})bis(\s+)")


def _fix_camelcase_compounds(text: str, st: _NormStats) -> str:
    if not text:
        return ""

    def repl(m: re.Match) -> str:
        st.camel_hyphen_fixed += 1
        return f"{m.group(1)}-{m.group(2)}"

    return _CAMEL_HYPHEN_RE.sub(repl, text)


def _fix_glued_conjunctions(text: str, st: _NormStats) -> str:
    if not text:
        return ""

    def repl_conj(m: re.Match) -> str:
        st.glued_fixed += 1
        return f"{m.group(1)} {m.group(2)}{m.group(3)}"

    def repl_bis(m: re.Match) -> str:
        st.glued_fixed += 1
        return f"{m.group(1)} bis{m.group(2)}"

    text = _GLUED_CONJ_RE.sub(repl_conj, text)
    text = _GLUED_BIS_RE.sub(repl_bis, text)
    return text


def _protect_abbrev_for_split(text: str, st: _NormStats) -> str:
    if not text:
        return ""

    t = text

    # Zahlenbereiche: "10./11." -> "10./11<DOT>"
    t, n = re.subn(r"\b(\d{1,2}\./\d{1,2})\.", r"\1" + _SPLIT_DOT, t)
    st.split_protected += n

    # Ordinale + Jahrhundert/Jh.
    t, n = re.subn(
        r"\b(\d{1,2})\.\s*(Jahrhundert|Jh\.?)",
        r"\1" + _SPLIT_DOT + r" \2",
        t,
        flags=re.IGNORECASE,
    )
    st.split_protected += n

    # Häufige Abkürzungen
    abbrev_patterns = [
        r"\bz\.\s*b\.",   # z. B.
        r"\bu\.\s*a\.",   # u. a.
        r"\bu\.\s*ä\.",   # u. Ä.
        r"\bv\.\s*chr\.", # v. Chr.
        r"\bn\.\s*chr\.", # n. Chr.
        r"\bu\.\s*u\.",   # u. U.
        r"\bz\.\s*t\.",   # z. T.
    ]

    for pat in abbrev_patterns:
        rx = re.compile(pat, re.IGNORECASE)

        def repl(m: re.Match) -> str:
            st.split_protected += 1
            return m.group(0).replace(".", _SPLIT_DOT)

        t = rx.sub(repl, t)

    # "u." als "und" in Aufzählungen: Schutz vor Satzsplit
    t, n = re.subn(r"\bu\.\s+(?=[A-ZÄÖÜ])", "u" + _SPLIT_DOT + " ", t)
    st.split_protected += n

    return t


def _restore_abbrev_after_split(text: str) -> str:
    if not text:
        return ""
    return text.replace(_SPLIT_DOT, ".")


def _strip_parentheticals(text: str, st: _NormStats) -> str:
    """
    Entfernt alle "(...)"-Inhalte inkl. Klammern (auch über mehrere Zeilen).
    """
    if not text:
        return ""
    t = text
    guard = 0
    while guard < 64:
        guard += 1
        t2, n = re.subn(r"\([^()]*\)", " ", t, flags=re.S)
        if n == 0:
            break
        st.parentheticals_stripped += n
        t = t2
    if t != text:
        t = re.sub(r"\s{2,}", " ", t).strip()
    return t


# ------------------------------------------------------------
# 1.2.6: Hyphen-Schutz & Rekonstruktion (Abkürzungs-Komposita)
# ------------------------------------------------------------

_ABBREV_PREFIXES = ("KI", "EU", "IT", "LLM", "AI")
# konservativ: nur diese Zielwörter werden bei ABBR+Word ohne Bindestrich repariert
_ABBREV_TARGET_WORDS = (
    "Verordnung", "System", "Systeme", "Modell", "Modelle", "Komponente", "Komponenten",
    "gestuetzte", "gestützte", "basiert", "basierte", "basierten", "basiertes", "basierten",
)

# "word- <space>word" -> "word-word" (ohne Stripping)
_HYPHEN_SPACE_RE = re.compile(r"\b([\wÄÖÜäöüß]+)-\s+([\wÄÖÜäöüß]+)\b")


def _fix_hyphen_spacing(text: str, st: _NormStats) -> str:
    if not text:
        return ""
    before = text
    text = _HYPHEN_SPACE_RE.sub(r"\1-\2", text)
    if text != before:
        st.hyphen_space_fixed += 1
    return text


def _restore_abbrev_hyphens(text: str, st: _NormStats) -> str:
    """
    Rekonstruiert fehlende Bindestriche bei Abkürzungs-Komposita:
      - "KIVerordnung" -> "KI-Verordnung" (Whitelist-basiert)
      - "KIgestuetzte" -> "KI-gestuetzte" (Whitelist-basiert)
      - "NonKISystem"  -> "Non-KI-System"
    Konservativ, um False-Positives zu vermeiden.
    """
    if not text:
        return ""

    before = text

    # NonKISystem / Non KI System / Non-KI System -> Non-KI-System (konservativ)
    # 1) NonKISystem
    text = re.sub(r"\bNonKISystem\b", "Non-KI-System", text)
    # 2) Non KI System (mit optionalen Hyphens/Spaces)
    text = re.sub(r"\bNon\s*-\s*KI\s*-\s*System\b", "Non-KI-System", text)
    text = re.sub(r"\bNon\s+KI\s+System\b", "Non-KI-System", text)

    # ABBR + TargetWord ohne Bindestrich: KIVerordnung, KIgestützte, EUVerordnung, LLMModell
    for abbr in _ABBREV_PREFIXES:
        for tw in _ABBREV_TARGET_WORDS:
            # exakt (Case-insensitive) aber Abk. soll in Originalform bleiben
            # -> wir setzen abbr wie im Pattern
            rx = re.compile(rf"\b{re.escape(abbr)}{re.escape(tw)}\b", re.IGNORECASE)
            text = rx.sub(lambda m: f"{abbr}-{m.group(0)[len(abbr):]}", text)

    if text != before:
        # grobe Zählung: 1 Tick wenn überhaupt Änderungen (Pareto)
        st.abbrev_hyphen_restored += 1
        if "Non-KI-System" in text and "Non-KI-System" not in before:
            st.nonki_restored += 1

    return text


def _strip_dangling_hyphen_tokens(text: str, st: _NormStats) -> str:
    """
    Letzter Rettungsanker:
    - entfernt Token-Endungen "-" wenn sie *dangling* sind, damit "haut-" nicht ins Vokab wandert.
    1.2.6: Schutz für Abkürzungs-Prefixe: "KI-" / "EU-" / "IT-" / "LLM-" bleiben erhalten.
    """
    if not text:
        return ""

    before = text

    # Match tokens like "haut-" "wort-" "abc-" but NOT "KI-" (all upper) or "EU-"
    # Wir strippen nur, wenn im Token mindestens ein Kleinbuchstabe vorkommt.
    def repl(m: re.Match) -> str:
        tok = m.group(1) or ""
        # wenn tok keinerlei Kleinbuchstaben enthält -> NICHT strippen
        has_lower = any(_is_lower(ch) for ch in tok)
        if not has_lower:
            return m.group(0)
        st.dangling_hyphen_stripped += 1
        return tok

    text = re.sub(r"\b([\wÄÖÜäöüß]+)-\b", repl, text)

    # (optional) cleanup: mehrfach-spaces durch ersetzte Tokens
    if text != before:
        text = re.sub(r"\s{2,}", " ", text).strip()

    return text


def _repair_sentence_split_abbrev_fragments(sentences: List[str], st: _NormStats) -> List[str]:
    """
    Repariert typische Wiki-Satzsplit-Artefakte bei Abkürzungen.

    Regex-Satzsplit trennt nach '.' + Whitespace auch dann, wenn es nur ein Abkürzungsfragment ist:
      "... , d." + "h. auf ..."   -> "... , d. h. auf ..."
      "... , d." + "auf ..."     -> "... , d. h. auf ..."
      "... , z." + "b. von ..."  -> "... , z. b. von ..."
      "... , z." + "von ..."     -> "... , z. b. von ..."
      "... o."  + "g. ..."       -> "... o. g. ..."
    """
    if not sentences:
        return []

    def starts_with_word(s: str, words: Tuple[str, ...]) -> bool:
        low = (s or "").lstrip().lower()
        return any(low == w or low.startswith(w + " ") for w in words)

    def strip_leading_abbrev_piece(s: str, piece: str) -> str:
        # piece: "h." / "b." / "g."
        low = (s or "").lstrip()
        rx = re.compile(rf"^{re.escape(piece)}\s*", re.IGNORECASE)
        return rx.sub("", low, count=1).strip()

    PREP_D = ("auf", "zu", "an", "bei", "in", "unter", "mit", "von")
    PREP_Z = ("von", "bei", "in", "zum", "zur", "für", "für", "auf", "als")

    out: List[str] = []
    i = 0
    n = len(sentences)

    while i < n:
        cur = (sentences[i] or "").strip()
        nxt = (sentences[i + 1] or "").strip() if i + 1 < n else ""

        # --- d. + h. ---
        if re.search(r"\bd\.\s*$", cur, flags=re.IGNORECASE) and nxt:
            if re.match(r"^h\.\s*", nxt, flags=re.IGNORECASE):
                rest = strip_leading_abbrev_piece(nxt, "h.")
                cur2 = re.sub(r"\bd\.\s*$", "d. h.", cur, flags=re.IGNORECASE)
                joined = (cur2 + (" " + rest if rest else "")).strip()
                out.append(joined)
                st.split_abbrev_joined += 1
                i += 2
                continue
            if starts_with_word(nxt, PREP_D):
                cur2 = re.sub(r"\bd\.\s*$", "d. h.", cur, flags=re.IGNORECASE)
                out.append((cur2 + " " + nxt).strip())
                st.split_abbrev_joined += 1
                i += 2
                continue

        # --- z. + b. ---
        if re.search(r"\bz\.\s*$", cur, flags=re.IGNORECASE) and nxt:
            if re.match(r"^b\.\s*", nxt, flags=re.IGNORECASE):
                rest = strip_leading_abbrev_piece(nxt, "b.")
                cur2 = re.sub(r"\bz\.\s*$", "z. b.", cur, flags=re.IGNORECASE)
                joined = (cur2 + (" " + rest if rest else "")).strip()
                out.append(joined)
                st.split_abbrev_joined += 1
                i += 2
                continue
            if starts_with_word(nxt, PREP_Z):
                cur2 = re.sub(r"\bz\.\s*$", "z. b.", cur, flags=re.IGNORECASE)
                out.append((cur2 + " " + nxt).strip())
                st.split_abbrev_joined += 1
                i += 2
                continue

        # --- o. + g. ---
        if re.search(r"\bo\.\s*$", cur, flags=re.IGNORECASE) and nxt:
            if re.match(r"^g\.\s*", nxt, flags=re.IGNORECASE):
                rest = strip_leading_abbrev_piece(nxt, "g.")
                cur2 = re.sub(r"\bo\.\s*$", "o. g.", cur, flags=re.IGNORECASE)
                joined = (cur2 + (" " + rest if rest else "")).strip()
                out.append(joined)
                st.split_abbrev_joined += 1
                i += 2
                continue

        out.append(cur)
        i += 1

    return out


def _split_head_suffix_from_compound(word: str) -> Optional[Tuple[str, str]]:
    """
    Versucht ein deutsches Kompositum in (Head, Suffix) zu splitten:
      "Gewebeabdrücken" -> ("Gewebe", "abdrücken")
      "Hauptsächlich"   -> None (kein sinnvoller Split)
    Heuristik:
    - Head beginnt typischerweise mit Großbuchstaben (Nomen).
    - Suffix beginnt typischerweise mit Kleinbuchstaben.
    - Wir bevorzugen den *längsten* Head, der diese Regeln erfüllt.
    """
    w = (word or "").strip()
    if len(w) < 6:
        return None

    if not _is_upper(w[0]):
        return None

    best: Optional[Tuple[str, str]] = None
    for i in range(len(w) - 2, 2, -1):
        head = w[:i]
        suf = w[i:]
        if len(suf) < 2:
            continue
        if not _is_lower(suf[0]):
            continue
        if len(head) < 3:
            continue

        ok_head = True
        for ch in head[1:]:
            if not (_is_lower(ch) or ch.isalpha()):
                ok_head = False
                break
            if _is_upper(ch):
                ok_head = False
                break
        if not ok_head:
            continue

        best = (head, suf)
        break

    return best


# ------------------------------------------------------------
# (NEW 1.2.2) Guard: Nur echte Dash-Enum-Fälle anfassen
# ------------------------------------------------------------
_DASH_ENUM_CANDIDATE_RE = re.compile(
    r"\b[\wÄÖÜäöüß]+-\s*(?:,|und|oder)\b",
    re.IGNORECASE
)


def _is_dash_enum_candidate(stems_block: str) -> bool:
    if not stems_block:
        return False
    return _DASH_ENUM_CANDIDATE_RE.search(stems_block) is not None


def _expand_dash_enumerations(text: str, st: _NormStats) -> str:
    """
    Repariert typische deutsche Kürzungs-Aufzählungen.

    Unterstützt u.a.:
    A) "Knochen-, Haut- und Gewebeabdrücken"
    B) "Knochen, Haut- und Gewebeabdrücken"    (ohne "-" am ersten Stamm)
    C) "Haut- und Gewebeabdrücken"             (2er-Fall)

    WICHTIG (1.2.2):
    - Expansion wird NUR ausgeführt, wenn im stems_block ein echtes Dash-Fragment vorkommt.
    """
    if not text:
        return ""

    parts = re.split(r"(?<=[.!?])\s+", text)
    out_parts: List[str] = []

    enum_re = re.compile(
        r"(?P<stems>(?:\b[\wÄÖÜäöüß]+(?:-)?\s*,\s*)*(?:\b[\wÄÖÜäöüß]+-?\s*(?:und|oder)\s+))(?P<last>\b[\wÄÖÜäöüß]+)\b"
    )
    stem_tok_re = re.compile(r"\b([\wÄÖÜäöüß]+)-?\b")

    for s in parts:
        s2 = s
        pos = 0
        guard_loops = 0

        while guard_loops < 64:
            guard_loops += 1
            m = enum_re.search(s2, pos)
            if not m:
                break

            stems_block = m.group("stems")
            last_word = m.group("last")

            if not _is_dash_enum_candidate(stems_block):
                pos = m.end()
                continue

            raw = stem_tok_re.findall(stems_block)
            stems = [t for t in raw if t.lower() not in ("und", "oder")]
            if not stems:
                pos = m.end()
                continue

            if " und " in stems_block:
                conj = "und"
            elif " oder " in stems_block:
                conj = "oder"
            else:
                conj = "und"

            suffix = None
            last_stem = stems[-1]
            if last_word.lower().startswith(last_stem.lower()):
                suffix = last_word[len(last_stem):]
                head = last_stem
                prev = stems[:-1]
            else:
                split = _split_head_suffix_from_compound(last_word)
                if not split:
                    pos = m.end()
                    continue
                head, suffix = split
                prev = stems[:]
                if prev and prev[-1].lower() == head.lower():
                    prev = prev[:-1]

            if not suffix or len(suffix.strip()) < 2:
                pos = m.end()
                continue

            expanded_left = ", ".join([p + suffix for p in prev]) if prev else ""
            expanded_right = head + suffix

            if expanded_left:
                expanded = f"{expanded_left} {conj} {expanded_right}"
            else:
                if stems:
                    expanded = f"{stems[0]}{suffix} {conj} {expanded_right}"
                else:
                    expanded = f"{expanded_right}"

            start, end = m.span()
            s2 = s2[:start] + expanded + s2[end:]
            st.enum_expanded += 1

            pos = 0

        out_parts.append(s2)

    return " ".join(out_parts)


def _dash_fragment_safety_net(text: str, st: _NormStats) -> str:
    """
    Safety-net gegen übrig gebliebene Tokens wie "haut-" in Fließtext.
    Fixiert v.a.:
      "Haut- und Gewebeabdrücken" -> "Hautabdrücken und Gewebeabdrücken"
    """
    if not text:
        return ""

    pat = re.compile(r"\b(?P<a>[\wÄÖÜäöüß]+)-\s+(?P<conj>und|oder)\s+(?P<b>[\wÄÖÜäöüß]+)\b")

    def repl(m: re.Match) -> str:
        a = m.group("a")
        conj = m.group("conj")
        b = m.group("b")

        split = _split_head_suffix_from_compound(b)
        if not split:
            return m.group(0)

        head, suffix = split
        st.safety_net += 1
        return f"{a}{suffix} {conj} {b}"

    return pat.sub(repl, text)


def normalize_import_text(text: str, config: dict, stats: _NormStats) -> str:
    if not text:
        return ""

    # Unicode konsolidieren (NFC), um Kombinationszeichen zu vermeiden
    text = unicodedata.normalize("NFC", text)

    if config.get("strip_invisible_unicode", True):
        text = _strip_invisible(text, stats)

    # erst Abkürzungen + Listen reparieren, solange Newlines noch existieren
    if config.get("repair_common_abbrev", True):
        text = _repair_common_abbrev(text, stats)

    if config.get("normalize_wiki_lists", True):
        text = _normalize_wiki_lists(text, stats)

    text = _unify_dashes(text, stats)

    # Newlines/CR entfernen (nach List-Fix)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # 1.2.6: "KI- Verordnung" etc.
    text = _fix_hyphen_spacing(text, stats)
    # 1.2.6: "KIVerordnung" / "NonKISystem"
    text = _restore_abbrev_hyphens(text, stats)

    # Punctuation-Cleanup vor Satzsplit
    text = _punctuation_cleanup(text, stats)

    # Glue-Fixes (z.B. "Texcocound", "Qinbis")
    if config.get("fix_glued_conjunctions", True):
        text = _fix_glued_conjunctions(text, stats)

    # CamelCase-Komposita (z.B. "MexikoStadt" -> "Mexiko-Stadt")
    if config.get("fix_camelcase_compounds", True):
        text = _fix_camelcase_compounds(text, stats)

    if config.get("expand_dash_enumerations", True):
        text = _expand_dash_enumerations(text, stats)

    if config.get("dash_fragment_safety_net", True):
        text = _dash_fragment_safety_net(text, stats)

    if config.get("strip_dangling_hyphen_tokens", True):
        text = _strip_dangling_hyphen_tokens(text, stats)

    # Finaler Space-Cleanup
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# Fetch: Wikipedia
# ============================================================

def fetch_wikipedia(topic: str, config: dict) -> str:
    url = "https://de.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": topic,
        "prop": "extracts",
        "explaintext": True,
        "formatversion": "2",
    }

    try:
        data = _http_get_json(
            url,
            params=params,
            headers={"User-Agent": config["user_agent"]},
            timeout=10,
        )
        text = data.get("query", {}).get("pages", [{}])[0].get("extract", "") or ""

        # Überschriften weg
        text = re.sub(r"={2,}.*?={2,}", "", text)

        stats = _NormStats()
        if config.get("strip_parentheticals", True):
            text = _strip_parentheticals(text, stats)
        if config.get("normalize_import_text", True):
            text = normalize_import_text(text, config, stats)
        else:
            text = text.replace("\n", " ")

        # Split-Protection (Abkürzungen/Zahlenbereiche)
        if config.get("protect_abbrev_for_split", True):
            text = _protect_abbrev_for_split(text, stats)

        # Satzsplit (grob)
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # Restore geschützte Abkürzungen
        sentences = [_restore_abbrev_after_split(s) for s in sentences]

        # Repair: Abkürzungs-Fragmente nach Satzsplit wieder zusammenführen
        sentences = _repair_sentence_split_abbrev_fragments(sentences, stats)

        if config.get("print_normalize_stats", True) and config.get("normalize_import_text", True):
            print(
                "      [norm] wiki:"
                f" strip={stats.strip_invisible}"
                f" dash={stats.dash_unified}"
                f" enum={stats.enum_expanded}"
                f" sn={stats.safety_net}"
                f" stripHy={stats.dangling_hyphen_stripped}"
                f" abbr={stats.abbrev_repaired}"
                f" bullets={stats.bullets_normalized}"
                f" punct={stats.punct_cleaned}"
                f" splitfix={stats.split_abbrev_joined}"
                f" hysp={stats.hyphen_space_fixed}"
                f" abhy={stats.abbrev_hyphen_restored}"
                f" nonki={stats.nonki_restored}"
                f" splitprot={stats.split_protected}"
                f" camel={stats.camel_hyphen_fixed}"
                f" glue={stats.glued_fixed}"
                f" paren={stats.parentheticals_stripped}"
            )

        limit = int(config.get("sentence_limit", config.get("line_limit", 10)) or 0)
        if limit <= 0:
            limit = int(config.get("line_limit", 10) or 10)

        cleaned = [s.strip() for s in sentences if len(s.strip()) > 10][:limit]
        return "\n".join(cleaned)

    except Exception:
        return ""


# ============================================================
# Fetch: Tatoeba (unstable)
# ============================================================

def fetch_tatoeba_api_new(count: int, config: dict) -> Optional[str]:
    url = "https://api.dev.tatoeba.org/unstable/sentences"
    headers = {"accept": "application/json", "User-Agent": config["user_agent"]}
    params = {"lang": "deu", "limit": int(count), "sort": "random"}

    try:
        print("      Probiere neue Tatoeba-API (unstable)...")
        data = _http_get_json(url, params=params, headers=headers, timeout=15) or {}
        items = data.get("data", data.get("results", [])) or []

        results: List[str] = []
        stats_all = _NormStats()

        for item in items:
            if isinstance(item, dict) and "text" in item:
                t = str(item["text"] or "").strip()
                if not t:
                    continue
                if config.get("normalize_import_text", True):
                    t = normalize_import_text(t, config, stats_all)
                results.append(t)

        if results:
            if (
                config.get("print_normalize_stats", True)
                and config.get("normalize_import_text", True)
            ):
                print(
                    "      [norm] tatoeba:"
                    f" strip={stats_all.strip_invisible}"
                    f" dash={stats_all.dash_unified}"
                    f" enum={stats_all.enum_expanded}"
                    f" sn={stats_all.safety_net}"
                    f" stripHy={stats_all.dangling_hyphen_stripped}"
                    f" abbr={stats_all.abbrev_repaired}"
                    f" bullets={stats_all.bullets_normalized}"
                    f" punct={stats_all.punct_cleaned}"
                    f" splitfix={stats_all.split_abbrev_joined}"
                    f" hysp={stats_all.hyphen_space_fixed}"
                    f" abhy={stats_all.abbrev_hyphen_restored}"
                    f" nonki={stats_all.nonki_restored}"
                    f" splitprot={stats_all.split_protected}"
                    f" camel={stats_all.camel_hyphen_fixed}"
                    f" glue={stats_all.glued_fixed}"
                )
            print(f"      {len(results)} Sätze via API geladen.")
            return "\n".join(results)

        raise ValueError("Keine Texte in der Antwort gefunden.")

    except Exception as e:
        print(f"      Tatoeba-API-Fehler: {e}")
        return None


# ============================================================
# Run Import
# ============================================================

def run_import() -> None:
    config = load_config()
    paths = setup_paths()

    force = bool(config.get("force_update", False))

    # ✅ USER: zuerst FORM
    print("--- FORM IMPORT ---")
    form_file = os.path.join(paths["form"], "01_tatoeba_alltag.txt")
    if (not os.path.exists(form_file)) or force:
        print(f" - Generiere {int(config.get('form_sentence_count', 0))} Sätze... (force={force})")
        sentences = fetch_tatoeba_api_new(int(config.get("form_sentence_count", 50)), config)

        if not sentences:
            print("      Nutze stattdessen Fallback...")
            fallback = "Guten Tag!\nWie geht es dir?\nDas ist ein Test."
            if config.get("normalize_import_text", True):
                stats = _NormStats()
                fallback = normalize_import_text(fallback, config, stats)
                if config.get("print_normalize_stats", True):
                    print(
                        "      [norm] fallback:"
                        f" strip={stats.strip_invisible}"
                        f" dash={stats.dash_unified}"
                        f" enum={stats.enum_expanded}"
                        f" sn={stats.safety_net}"
                        f" stripHy={stats.dangling_hyphen_stripped}"
                        f" abbr={stats.abbrev_repaired}"
                        f" bullets={stats.bullets_normalized}"
                        f" punct={stats.punct_cleaned}"
                        f" splitfix={stats.split_abbrev_joined}"
                        f" hysp={stats.hyphen_space_fixed}"
                        f" abhy={stats.abbrev_hyphen_restored}"
                        f" nonki={stats.nonki_restored}"
                        f" splitprot={stats.split_protected}"
                        f" camel={stats.camel_hyphen_fixed}"
                        f" glue={stats.glued_fixed}"
                    )
            sentences = fallback

        with open(form_file, "w", encoding="utf-8") as f:
            f.write(sentences)
    else:
        print(" - Form aktuell.")

    # ✅ dann KNOWLEDGE
    print("\n--- KNOWLEDGE IMPORT ---")
    for i, topic in enumerate(config.get("knowledge_topics", []) or [], 1):
        filename = f"{i:02d}_{clean_filename(topic)}.txt"
        file_path = os.path.join(paths["knowledge"], filename)

        if (not os.path.exists(file_path)) or force:
            print(f" - [{i:02d}] Lade: {topic}... (force={force})")
            content = fetch_wikipedia(topic, config)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            print(f" - [{i:02d}] Überspringe: {topic}")

    print("\n--- IMPORT ABGESCHLOSSEN ---")


if __name__ == "__main__":
    run_import()

# ============================================================
# <<< END FILE: llm_training/import/import_training_data.py
# ROWCOUNT_OLD: 787
# ROWCOUNT_NEW: 901
# ============================================================
