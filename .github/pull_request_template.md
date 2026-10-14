## What changed and why

<!-- One or two sentences. Link the phase or issue. -->

## Review checklist

- [ ] Tests added or updated for new logic (`pytest -q` passes locally)
- [ ] `ruff check src scripts tests notebooks` is clean
- [ ] Notebook `.py` sources changed? Rebuilt with `python scripts/build_notebooks.py` (and executed on real data if results changed)
- [ ] No raw TLC rows, credentials or local paths in code, notebooks or outputs
- [ ] Numbers quoted in README / memo come from `reports/metrics/*.json`, not typed by hand
- [ ] Model artifact changed? `model_version` bumped by `scripts/export_model.py`, and the model card updated
- [ ] Pay changes still respect C1 (floor ≤ pay ≤ 1.3 × floor) — covered by `tests/test_api.py`
