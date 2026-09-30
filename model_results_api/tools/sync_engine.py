"""Regenerate `model_results_api/engine/` from the `src/` originals.

The API ships as a self-contained service — its own requirements.txt, its own
process, no dependency on the analysis tree — which is why these modules are
vendored rather than imported. The price of vendoring is drift, and a stale copy
of a *feature builder* does not fail loudly: it quietly computes features the
model was never trained on, and every response stays a plausible-looking number.

So the copies are generated, never hand-edited, and `test_engine_sync.py` turns
the suite red the moment they stop matching. After changing any engine module in
`src/`, run:

    py model_results_api/tools/sync_engine.py

The transform is deliberately tiny: `src/` modules are written to be runnable as
top-level scripts and reach their siblings through a sys.path insertion, while
the vendored package reaches them through relative imports. Nothing else differs,
so nothing else may be rewritten here.
"""
from __future__ import annotations

import re
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = API_ROOT.parent
SRC_DIR = WORKSPACE_ROOT / "src"
ENGINE_DIR = API_ROOT / "engine"

MODULES = ("article_prospects", "prospect_features", "prospect_model")

_PATH_HACK = "sys.path.insert(0, str(Path(__file__).resolve().parent))"
_SIBLING_IMPORT = re.compile(r"^import (\w+) as (\w+)$")


def vendored_source(text: str) -> str:
    """Rewrite one src/ engine module into its vendored package form."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line == "import sys":
            i += 1
            continue
        if line == _PATH_HACK:
            i += 1
            # The hack is followed by a blank separator that has nothing left to
            # separate once the hack is gone.
            if i < len(lines) and lines[i] == "":
                i += 1
            continue
        match = _SIBLING_IMPORT.match(line)
        if match and match.group(1) in MODULES:
            out.append(f"from . import {match.group(1)} as {match.group(2)}")
            i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def expected() -> dict[str, str]:
    return {name: vendored_source((SRC_DIR / f"{name}.py").read_text(encoding="utf-8"))
            for name in MODULES}


def main() -> int:
    changed = []
    for name, text in expected().items():
        target = ENGINE_DIR / f"{name}.py"
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8")
            changed.append(name)
    print(f"synced {len(changed)} module(s): {', '.join(changed) or 'none, already current'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
