# -*- coding: utf-8 -*-
import importlib.util
import os
import re as _re

import requests

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
IMPORTER_PATH = os.path.join(THIS_DIR, "import_training_data.py")

def stutter_hits(s: str) -> int:
    if not s:
        return 0
    pat = _re.compile(r"([A-Za-zÄÖÜäöüß]{2,3})\1{2,}")
    return len(list(pat.finditer(s)))

def load_importer():
    spec = importlib.util.spec_from_file_location("import_training_data", IMPORTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

def fetch_raw_deutsche_sprache(user_agent="MobileBot/1.0"):
    url = "https://de.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": "Deutsche Sprache",
        "prop": "extracts",
        "explaintext": True,
        "formatversion": "2",
    }
    r = requests.get(url, params=params, headers={"User-Agent": user_agent}, timeout=15)
    data = r.json() or {}
    return (data.get("query", {}).get("pages", [{}])[0].get("extract", "") or "")

def main():
    mod = load_importer()
    cfg = {}
    try:
        cfg = mod.load_config()
    except Exception:
        cfg = {"user_agent": "MobileBot/1.0"}

    raw = fetch_raw_deutsche_sprache(cfg.get("user_agent", "MobileBot/1.0"))
    print("RAW stutter hits:", stutter_hits(raw))

    # --- patch re.sub im Importer, um die Regel zu finden, die Stottern erzeugt ---
    real_sub = _re.sub
    def sub_wrapper(pattern, repl, string, count=0, flags=0):
        before_hits = stutter_hits(string)
        out = real_sub(pattern, repl, string, count=count, flags=flags)
        after_hits = stutter_hits(out)
        if after_hits > before_hits:
            # pattern kann str oder compiled sein
            ptxt = getattr(pattern, "pattern", pattern)
            print("\n[STUTTER +] pattern:", ptxt)
            print("  hits:", before_hits, "->", after_hits)
            # kleiner Kontext-Dump
            idx = out.find("chchch")  # heuristisch
            if idx != -1:
                a = max(0, idx - 80)
                b = min(len(out), idx + 160)
                print("  ctx:", out[a:b].replace("\n", " "))
        return out

    # importer nutzt "re" global → wir patchen dessen sub
    mod.re.sub = sub_wrapper

    # 1. Normalize
    st1 = mod._NormStats()
    norm1 = mod.normalize_import_text(raw, cfg, st1)
    print("\nNORM1 stutter hits:", stutter_hits(norm1))

    # 2. Normalize nochmal (Idempotenz)
    st2 = mod._NormStats()
    norm2 = mod.normalize_import_text(norm1, cfg, st2)
    print("NORM2 stutter hits:", stutter_hits(norm2))

    print("\nDONE. Wenn oben [STUTTER +] geloggt wurde, ist das die schuldige Regex-Regel.")

if __name__ == "__main__":
    main()
