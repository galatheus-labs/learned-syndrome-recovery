# Gauge Paper — internal working folder

Private working directory for the third paper, the one that unifies the
QEC and compression papers under a gauge-theoretic / fracton lens. **Do
not move these files into `papers/`** until the two small papers
(`qec_arxiv/` and `compression_arxiv/`) have shipped — they avoid
foreshadowing the unification thesis to keep the lede here.

Contents:

- `PAPER_DRAFT.md` — working outline + main theorem + planned
  experiments. Read this first.
- `verify_fracton_theorem.py` — pure-Python F₂ rank verifier for
  the main theorem on `n ∈ {4, 6, 8}`. No deps.
- `run_log.txt` — captured output from the verifier; pasted into the
  empirical-verification section of the draft.

To re-run the verification:

```
python3 verify_fracton_theorem.py
```

(takes ~20 s on n≤8; pure Python F₂ Gaussian elimination)
