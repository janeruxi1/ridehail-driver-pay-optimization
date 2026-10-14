"""Build the .ipynb notebooks from their paired .py sources, and optionally execute them.

The .py file (percent format: ``# %%`` code cells, ``# %% [markdown]`` text cells) is the
source of truth: CI runs it, review reads it. The .ipynb is generated from it so the two
cannot drift, then executed on the real data so GitHub shows every table and figure.

Usage (from the project root):
    python scripts/build_notebooks.py                 # build all, keep existing outputs if the code is unchanged
    python scripts/build_notebooks.py --execute       # build and run all on the real data
    python scripts/build_notebooks.py --execute 02_position_effects
    python scripts/build_notebooks.py --check         # verify every .ipynb matches its .py; write nothing

Outputs contain only aggregates (counts, rates, model coefficients, figures), never rows of
the raw data, which may not be redistributed. Execution refuses to run on the synthetic
table so a committed notebook always shows real results.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
KERNEL = {"display_name": "Python 3", "language": "python", "name": "python3"}


def parse_percent(text: str) -> list[dict]:
    """Split a percent-format script into notebook cells."""
    cells: list[dict] = []
    kind, lines = None, []

    def flush():
        if kind is None:
            return
        body = list(lines)
        while body and not body[-1].strip():
            body.pop()
        while body and not body[0].strip():
            body.pop(0)
        if kind == "markdown":
            body = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in body]
        if body:
            cells.append({"cell_type": kind, "source": "\n".join(body)})

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        else:
            if kind is None:
                kind = "code"
            lines.append(line)
    flush()
    return cells


def to_notebook(cells: list[dict]) -> dict:
    out = []
    for i, c in enumerate(cells):
        cell = {"cell_type": c["cell_type"], "id": str(i), "metadata": {}, "source": c["source"].splitlines(keepends=True)}
        if c["cell_type"] == "code":
            cell.update(execution_count=None, outputs=[])
        out.append(cell)
    return {"cells": out, "metadata": {"kernelspec": KERNEL, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def sources(nb: dict) -> list[tuple[str, str]]:
    return [(c["cell_type"], "".join(c["source"]).rstrip()) for c in nb["cells"]]


def build(name: str, execute: bool) -> str:
    src = NOTEBOOKS / f"{name}.py"
    dst = NOTEBOOKS / f"{name}.ipynb"
    fresh = to_notebook(parse_percent(src.read_text(encoding="utf-8")))
    if not execute:
        if dst.exists():
            old = json.loads(dst.read_text(encoding="utf-8"))
            if sources(old) == sources(fresh):
                return "unchanged"
        with open(dst, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(fresh, indent=1, ensure_ascii=False) + "\n")
        return "built (no outputs; run with --execute)"
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.reads(json.dumps(fresh), as_version=4)
    NotebookClient(nb, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": str(NOTEBOOKS)}}).execute()
    for cell in nb.cells:  # drop timestamps so re-running does not create noise in git
        cell.metadata.pop("execution", None)
    nb.metadata.pop("widgets", None)
    nb.metadata["language_info"] = {"name": "python"}
    with open(dst, "w", encoding="utf-8", newline="\n") as fh:  # LF on every OS
        fh.write(nbformat.writes(nb))
    return "built and executed"


def check() -> list[str]:
    problems = []
    for src in sorted(NOTEBOOKS.glob("[0-9]*.py")):
        dst = src.with_suffix(".ipynb")
        if not dst.exists():
            problems.append(f"{dst.name} missing")
            continue
        fresh = to_notebook(parse_percent(src.read_text(encoding="utf-8")))
        if sources(json.loads(dst.read_text(encoding="utf-8"))) != sources(fresh):
            problems.append(f"{dst.name} does not match {src.name}; run scripts/build_notebooks.py")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="notebook names without extension (default: all)")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if args.check:
        problems = check()
        print("\n".join(problems) if problems else "All notebooks match their .py sources.")
        return 1 if problems else 0
    if args.execute:
        sys.path.insert(0, str(ROOT / "src"))
        from ridepay.data import real_data_available

        if not real_data_available():
            print("Refusing to execute: the real data is not available (or RIDEPAY_SYNTHETIC is set).\n"
                  "Executed notebooks must show real results. Run scripts/build_tables.py first.")
            return 1
        os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")
        if sys.platform == "win32":  # pyzmq needs the selector event loop on Windows; avoids a noisy warning
            import asyncio

            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    names = args.names or [p.stem for p in sorted(NOTEBOOKS.glob("[0-9]*.py"))]
    for name in names:
        print(f"{name}: {build(name, args.execute)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
