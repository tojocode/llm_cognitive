# -*- coding: utf-8 -*-
# ============================================================
# Attention-Cluster LLM – Wikipedia → Core Knowledge Preprocessor
# Datei: llm_training/train/wiki_to_core.py
#
# VERSION: 1.0.4
# STATUS: STABIL
#
# ROLLE:
# - Wandelt Wikipedia-Import in "Core-taugliche" Knowledge-Zeilen um
# - Setzt Marker heuristisch (Single source of truth: marker_table.py)
#
# USER-POLICY (Fix):
# - Erster Satz immer <def>
# - Ersten Satz NICHT splitten
# - Alles in Klammern "(...)" entfernen
#
# INPUT:
# - llm_training/data/knowledge/01_core/**/*.txt
#
# OUTPUT:
# - llm_training/data/knowledge/02_core/**/*.txt
#
# ÄNDERUNGEN:
# 1.0.4 – FIX: <not> nur noch bei "starker" Negation (ist/sind/war/wird ... nicht)
#         FIX: "weiche" Nicht-Phrasen (noch nicht / nicht mehr / nicht allein / nicht nur / ...)
#              markieren NICHT mehr als <not> (fallen auf <prop>/<soft> zurück).
# 1.0.3 – FIX: Marker-Priorität: <class>/<def> vor <soft> (Wiki-Definitionen stabil)
#         FIX: "bezeichnung" nicht mehr als <soft>-Trigger (sonst False-Soft)
#         ADD: <class>-Trigger für "ist (außerdem) die bezeichnung für"
#         FIX: Default-IN_ROOT auf 01_core (Fallback: 01_wikipedia)
# 1.0.2 – Initial stabil
#
# ============================================================

from __future__ import annotations

import os
import sys
import re
from typing import List, Tuple, Optional


# ------------------------------------------------------------
# Projekt-Root robust setzen (iOS/Pythonista-safe)
# ------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))                 # .../llm_training/train
LLM_TRAINING_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))      # .../llm_training
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))    # .../<project root>

for p in (PROJECT_ROOT, LLM_TRAINING_DIR, THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    os.chdir(PROJECT_ROOT)
except Exception:
    pass


# ------------------------------------------------------------
# Marker Table (Single Source of Truth)
# ------------------------------------------------------------
try:
    from llm_training.train.marker_table import all_markers  # type: ignore
except Exception:
    from marker_table import all_markers  # type: ignore


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
# ✅ Dein Pipeline-Standard: Import schreibt nach 01_core
IN_ROOT = os.path.join(PROJECT_ROOT, "llm_training", "data", "knowledge", "01_wikipedia-import")

# Fallback (alte Struktur, falls vorhanden)
ALT_IN_ROOT = os.path.join(PROJECT_ROOT, "llm_training", "data", "knowledge", "01_wikipedia-import")

# ✅ Output immer nach 02_core (du verschiebst danach 02 → 01)
OUT_ROOT = os.path.join(PROJECT_ROOT, "llm_training", "data", "knowledge", "01_core")


# ------------------------------------------------------------
# Config (Pareto)
# ------------------------------------------------------------
MIN_CHARS_SENT = 12
MIN_WORDS_SENT = 4
MAX_WORDS_HARD = 40         # zu lang -> <soft>
MAX_SENT_SPLIT = 50         # Safety für extrem lange Absätze
LOWERCASE_OUTPUT = True

_END_CHARS = ".!?"

# Abkuerzungen / Satz-Ende-Fallen (wichtig: erster Satz darf NICHT an "z." etc. enden)
_ABBREV = {
    "z.b.", "u.a.", "d.h.", "bzw.", "vgl.", "evtl.", "u.ä.",
    "i.d.r.", "i.e.", "e.g.",
    "v.", "ca.", "nr.", "dr.", "prof.",
    # sehr haeufige 1-letter Segmente im Wiki-Text (d. auf / z. b. / u. a.)
    "z.", "b.", "u.", "a.", "d.", "i.",
}

# ⚠️ "bezeichnung" ist in Wiki häufig Teil von Definition/Klassifikation
# und darf NICHT <soft> erzwingen.
_SOFT_TRIGGERS = (
    "altgriech", "latein", "englisch", "begriff", "genannt",
    "geprägt", "rockefeller", "stiftung", "dartmouth", "forschungsprojekt",
    "jahrhundert", "v. chr", "n. chr", "sommer", "wissenschaftler",
    "historisch", "etymologie", "siehe auch", "diskurs", "zuordnung",
)

_DEF_TRIGGERS = (
    "bezeichnet", "ist die wissenschaft", "ist die lehre", "ist ein begriff",
    "wird definiert", "definiert als", "steht fuer", "steht für",
)

_CLASS_TRIGGERS = (
    "ist ein teilgebiet", "ist ein teilbereich", "ist ein teil des",
    "ist eine form von", "gehört zu", "gehoert zu", "zählt zu", "zaehlt zu",
    "ist eine art von", "ist eine gruppe von",
    # ✅ Wiki-typisch:
    "ist die bezeichnung für", "ist außerdem die bezeichnung für", "ist auch die bezeichnung für",
)

_CAUSE_TRIGGERS = ("weil", "deshalb", "darum", "daher", "denn", "sodass", "so dass", "dadurch")

_RE_JUNK_LINE = re.compile(r"^\s*[\-\*\u2022]+\s*")
_RE_MULTI_WS = re.compile(r"\s+")
_RE_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")

_EDGE_STRIP = "\"'„“‚‘()[]{}"

# Klammern entfernen: ( ... ) — iterativ, nicht-nested (Pareto)
_RE_PARENS = re.compile(r"\([^)]*\)")

# "weiche" Nicht-Phrasen: NICHT als <not> markieren (sondern <prop>/<soft>)
_NOT_SOFT_PHRASES = (
    "noch nicht",
    "nicht mehr",
    "nicht nur",
    "nicht allein",
    "nicht alleine",
    "nicht ausschließlich",
    "nicht unbedingt",
    "nicht immer",
    "nicht selten",
    "nicht sehr",
    "nicht ganz",
    "nicht vollständig",
    "nicht transparent",
    "nicht einsehbar",
    "nicht bekannt",
    "nicht klar",
    "nicht eindeutig",
)

# "starke" Negation: <not> nur wenn diese Struktur vorliegt (Pareto)
_RE_STRONG_NOT = re.compile(
    r"\b(ist|sind|war|waren|wird|werden|bleibt|bleiben|gilt|gelten|sei|seien)\s+nicht\b",
    re.IGNORECASE,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _norm_ws(s: str) -> str:
    s = (s or "").strip()
    s = _RE_MULTI_WS.sub(" ", s)
    return s.strip()


def _strip_edges(s: str) -> str:
    t = (s or "").strip()
    t = t.strip(_EDGE_STRIP).strip()
    return t


def remove_parentheses(text: str) -> str:
    """
    Entfernt grundsätzlich ALLES in runden Klammern "(...)"
    (nicht-nested, iterativ – Pareto/robust).
    """
    s = (text or "")
    for _ in range(10):
        s2 = _RE_PARENS.sub("", s)
        if s2 == s:
            break
        s = s2
    return _norm_ws(s)


def _looks_like_abbrev(prev: str, cur: str) -> bool:
    a = (prev or "").strip().lower()
    b = (cur or "").strip().lower()
    if not a or not b:
        return False

    combo = (a + b).replace(" ", "")
    if combo in _ABBREV:
        return True

    if a.endswith(".") and b.endswith(".") and (a + " " + b).replace(" ", "") in _ABBREV:
        return True

    return False


def _token_before_dot(s: str, dot_idx: int) -> str:
    j = dot_idx - 1
    while j >= 0 and s[j].isspace():
        j -= 1
    k = j
    while k >= 0 and (s[k].isalnum() or s[k] in "äöüÄÖÜß"):
        k -= 1
    tok = s[k + 1 : j + 1].strip().lower()
    return tok


def _next_nonspace(s: str, i: int) -> str:
    j = i
    n = len(s)
    while j < n and s[j].isspace():
        j += 1
    return s[j] if j < n else ""


def _is_first_sentence_end(s: str, i: int) -> bool:
    """
    Heuristik für erstes Satzende (ohne "splitten"):
    - akzeptiert .!? nur, wenn danach Whitespace/Ende ist
    - bei '.' zusätzlich: keine Abkürzung / kein 1-letter-Kettchen (z. b.) / keine Dezimalzahl
    """
    ch = s[i]
    n = len(s)

    nxt = s[i + 1] if (i + 1) < n else ""
    if nxt and (not nxt.isspace()):
        return False

    if ch in "!?":
        return True

    if ch != ".":
        return True

    prevc = s[i - 1] if i - 1 >= 0 else ""
    nextc = _next_nonspace(s, i + 1)
    if prevc.isdigit() and nextc.isdigit():
        return False

    tok = _token_before_dot(s, i)
    if tok and (tok + ".") in _ABBREV:
        return False

    if len(tok) == 1 and tok.isalpha():
        if nextc.isalpha():
            return False

    return True


def split_sentences(text: str) -> List[str]:
    """
    iOS-safe Satzsplitter (für Sätze NACH dem ersten Satz).
    """
    s = _norm_ws(text)
    if not s:
        return []

    out: List[str] = []
    buf: List[str] = []
    token_buf: List[str] = []

    def flush_token():
        nonlocal token_buf
        token_buf = []

    def flush_sentence():
        nonlocal buf
        sent = _norm_ws("".join(buf))
        if sent:
            out.append(sent)
        buf = []

    n = len(s)
    for i, ch in enumerate(s):
        buf.append(ch)

        if ch.isspace():
            flush_token()
        else:
            token_buf.append(ch)

        if ch in _END_CHARS:
            flush_token()

            nxt = s[i + 1] if (i + 1) < n else ""
            if nxt and (not nxt.isspace()):
                continue

            tail = _norm_ws("".join(buf))
            tail_tokens = tail.split(" ")
            prev = tail_tokens[-2].lower() if len(tail_tokens) >= 2 else ""
            cur = tail_tokens[-1].lower() if len(tail_tokens) >= 1 else ""
            if _looks_like_abbrev(prev, cur):
                continue

            flush_sentence()
            if len(out) >= MAX_SENT_SPLIT and n > 2000:
                break

    rest = _norm_ws("".join(buf))
    if rest:
        out.append(rest)

    return out


def extract_first_sentence_no_split(text: str) -> Tuple[str, str]:
    """
    USER-POLICY:
    - erster Satz NICHT splitten/fragmentieren
    - aber wir müssen das Satzende finden (Wiki: i.d.R. erster Punkt nach echter Definition)
    """
    s = _norm_ws(text)
    if not s:
        return "", ""

    n = len(s)
    for i, ch in enumerate(s):
        if ch in _END_CHARS:
            if _is_first_sentence_end(s, i):
                first = _norm_ws(s[: i + 1])
                rest = _norm_ws(s[i + 1 :])
                return first, rest

    return s, ""


def normalize_sentence(sent: str) -> str:
    s = _norm_ws(sent)
    s = _RE_JUNK_LINE.sub("", s).strip()
    s = remove_parentheses(s)
    s = _strip_edges(s)
    s = s.replace(" ,", ",").replace(" ;", ";").replace(" :", ":").strip()
    s = s.replace("„", "\"").replace("“", "\"").replace("‚", "'").replace("‘", "'")
    s = _norm_ws(s)
    return s


def is_usable_sentence(sent: str) -> bool:
    s = (sent or "").strip()
    if len(s) < MIN_CHARS_SENT:
        return False
    words = [w for w in s.split(" ") if w]
    if len(words) < MIN_WORDS_SENT:
        return False

    bad = sum(1 for ch in s if ch in "<>[]")
    if bad >= 2:
        return False

    return True


def detect_topic_from_text(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    for line in text.splitlines():
        t = _norm_ws(line)
        if not t:
            continue
        cut = t
        for sep in ("(", ",", "–", "-", ":", ";"):
            if sep in cut:
                cut = cut.split(sep, 1)[0].strip()
        cut = cut.strip()
        if cut and len(cut.split()) <= 6:
            return cut.lower()
        w0 = t.split(" ")[0].strip().lower()
        if w0 and len(w0) >= 3:
            return w0
        break
    return None


def _should_use_not(low: str) -> bool:
    if " nicht " not in low:
        return False

    for ph in _NOT_SOFT_PHRASES:
        if ph in low:
            return False

    return _RE_STRONG_NOT.search(low) is not None


def choose_marker(sent: str, *, topic: Optional[str], allowed_markers: set) -> str:
    """
    Marker-Priorität (Pareto, Wiki-tauglich):
    NEG → (starkes) NOT → CAUSE → CLASS → DEF → SOFT → PROP → fallback
    """
    s = (sent or "").strip()
    low = s.lower()

    # Negation (stark: "kein/e...")
    if any(x in low for x in (" kein ", " keine ", " keinen ", " keinem ", " keiner ")):
        if "<neg>" in allowed_markers:
            return "<neg>"

    # NOT (reduziert: nur "starke" Negation)
    if _should_use_not(low):
        if "<not>" in allowed_markers:
            return "<not>"

    # Kausalität
    if any(k in low for k in _CAUSE_TRIGGERS):
        if "<cause>" in allowed_markers:
            return "<cause>"

    # Klassifikation / Definition (muss VOR <soft> kommen!)
    if any(k in low for k in _CLASS_TRIGGERS):
        if "<class>" in allowed_markers:
            return "<class>"

    if any(k in low for k in _DEF_TRIGGERS):
        if "<def>" in allowed_markers:
            return "<def>"

    if topic:
        if low.startswith(topic + " ist ") or low.startswith(topic + " bezeichnet "):
            if "<def>" in allowed_markers:
                return "<def>"

    # Soft (Wiki-Historie/Etymologie/Meta)
    wc = len([w for w in low.split(" ") if w])
    if wc > MAX_WORDS_HARD:
        if "<soft>" in allowed_markers:
            return "<soft>"

    if any(k in low for k in _SOFT_TRIGGERS) or _RE_YEAR.search(low):
        if "<soft>" in allowed_markers:
            return "<soft>"

    # Default: Proposition
    if "<prop>" in allowed_markers:
        return "<prop>"

    return "<soft>" if "<soft>" in allowed_markers else "<def>"


def preprocess_text(text: str) -> List[Tuple[str, str]]:
    """
    Returns: List[(marker, sentence)]
    USER-POLICY:
    - first sentence ALWAYS <def>
    - first sentence NOT split
    - remove parentheses globally
    """
    allowed = set(all_markers())
    topic = detect_topic_from_text(text)

    lines: List[str] = []
    for raw in (text or "").splitlines():
        s = _norm_ws(raw)
        if not s:
            continue
        lines.append(s)

    joined = " ".join(lines)
    joined = remove_parentheses(joined)

    first_raw, rest = extract_first_sentence_no_split(joined)

    out: List[Tuple[str, str]] = []

    # First sentence: forced <def>, no heuristic, no split
    first = normalize_sentence(first_raw)
    if first and is_usable_sentence(first):
        if LOWERCASE_OUTPUT:
            first = first.lower()
        out.append(("<def>" if "<def>" in allowed else "<prop>", first))

    # Remaining sentences
    if rest:
        sents = split_sentences(rest)
        for sent in sents:
            ns = normalize_sentence(sent)
            if not is_usable_sentence(ns):
                continue
            mk = choose_marker(ns, topic=topic, allowed_markers=allowed)
            if LOWERCASE_OUTPUT:
                ns = ns.lower()
            out.append((mk, ns))

    return out


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def relpath_under(root: str, path: str) -> str:
    rp = os.path.relpath(path, root)
    return rp.replace("\\", "/")


def _resolve_in_root() -> str:
    if os.path.isdir(IN_ROOT):
        return IN_ROOT
    if os.path.isdir(ALT_IN_ROOT):
        return ALT_IN_ROOT
    return IN_ROOT


def main() -> None:
    in_root = _resolve_in_root()

    print("=== Wikipedia → Core Knowledge Preprocessor ===")
    print(f"IN : {in_root}")
    print(f"OUT: {OUT_ROOT}")

    if not os.path.isdir(in_root):
        raise RuntimeError(f"❌ Input-Ordner nicht gefunden: {in_root}")

    _ensure_dir(OUT_ROOT)

    total_in_files = 0
    total_out_files = 0
    total_in_chars = 0
    total_out_lines = 0

    per_file_stats: List[Tuple[str, int]] = []

    for root, _, files in os.walk(in_root):
        for fn in files:
            if not fn.endswith(".txt"):
                continue

            total_in_files += 1
            in_path = os.path.join(root, fn)
            rel = relpath_under(in_root, in_path)
            out_path = os.path.join(OUT_ROOT, rel)
            out_dir = os.path.dirname(out_path)
            _ensure_dir(out_dir)

            try:
                with open(in_path, "r", encoding="utf-8") as f:
                    raw = f.read()
            except Exception as e:
                print(f"⚠️  Skip (read error): {rel} ({e})")
                continue

            total_in_chars += len(raw)

            pairs = preprocess_text(raw)
            if not pairs:
                print(f"⚠️  Skip (no usable sents): {rel}")
                continue

            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    for mk, s in pairs:
                        f.write(f"{mk} {s}\n")
            except Exception as e:
                print(f"❌ Write error: {rel} ({e})")
                continue

            total_out_files += 1
            total_out_lines += len(pairs)
            per_file_stats.append((rel, len(pairs)))

            if total_out_files % 25 == 0:
                print(f"... processed: {total_out_files} files")

    print("\n=== DONE ===")
    print(f"input_files : {total_in_files}")
    print(f"output_files: {total_out_files}")
    print(f"input_chars : {total_in_chars}")
    print(f"output_lines: {total_out_lines}")

    per_file_stats.sort(key=lambda x: x[1], reverse=True)
    if per_file_stats:
        print("\nTop outputs:")
        for rel, n in per_file_stats[:10]:
            print(f"  {n:5d}  {rel}")

    print("\n✅ Output liegt unter: llm_training/data/knowledge/02_core")


if __name__ == "__main__":
    main()

# ============================================================
# <<< END FILE: llm_training/train/wiki_to_core.py
# ROWCOUNT_OLD: 582
# ROWCOUNT_NEW: 602
# ============================================================
