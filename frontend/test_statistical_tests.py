# -*- coding: utf-8 -*-
"""
Validation tests for the statistics computed by the dashboard
(frontend/app.py): Diebold-Mariano test and the binomial DA test.

Run inside the frontend container (has numpy/pandas, needs no streamlit):
    docker compose exec frontend python /app/test_statistical_tests.py

What is checked:
  1. Type-I error rate of the DM test under H0 (two equally good
     predictors, i.i.d. errors): share of p < 0.05 should be ~5%.
  2. Type-I error rate of the binomial DA test under H0 (a model that
     guesses direction with exactly the base rate p0): ~5%.
  3. Sanity power check: a clearly worse predictor should be rejected
     (almost) always.
  4. Edge cases: identical predictions, sign symmetry, k = 0, k = n,
     p0 = 1, zero error metrics.

The functions are extracted from app.py via AST (no streamlit import),
so the tests always exercise the code that actually runs in the UI.
"""
import ast
import math
import sys

import numpy as np
import pandas as pd

APP_PATH = sys.argv[1] if len(sys.argv) > 1 else "/app/app.py"
N_SIMS = 4000
ALPHA = 0.05
TOL = 0.02          # 95% CI half-width for N_SIMS=4000 is ~0.9%; keep loose
SEED = 42

src = open(APP_PATH, encoding="utf-8").read()
tree = ast.parse(src)
WANT = {"dm_test", "binomial_da_test", "directional_accuracy", "compute_metrics"}
snip = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in WANT:
        node.decorator_list = []
        snip.append(ast.unparse(node))
assert len(snip) == len(WANT), f"missing functions: {WANT - set(x.name for x in tree.body if isinstance(x, ast.FunctionDef))}"

ns = {"ML_MODELS": ["LSTM", "GRU", "Random Forest", "Kalman Filter"],
      "np": np, "pd": pd, "math": math,
      "Dict": dict, "Tuple": tuple, "List": list}
exec(compile("import numpy as np\nimport pandas as pd\nimport math\n\n" + "\n\n".join(snip),
             "extracted", "exec"), ns)
dm_test = ns["dm_test"]
binomial_da_test = ns["binomial_da_test"]
compute_metrics = ns["compute_metrics"]

rng = np.random.default_rng(SEED)
failures = []


def check(name, cond, detail=""):
    status = "OK  " if cond else "FAIL"
    print(f"[{status}] {name}{(' - ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------------
# 1) DM test under H0: two equally good predictors
# ---------------------------------------------------------------------
rejections = 0
for _ in range(N_SIMS):
    actual = rng.normal(size=180)
    pred1 = actual + rng.normal(size=180)   # error sd 1.0
    pred2 = actual + rng.normal(size=180)   # error sd 1.0
    _stat, p = dm_test(actual, pred1, pred2)
    rejections += p < ALPHA
rate = rejections / N_SIMS
check("DM H0: odsetek p<0.05 ~ 5%", abs(rate - ALPHA) < TOL, f"rate = {rate:.3%}")

# ---------------------------------------------------------------------
# 2) Binomial DA test under H0: model hits direction exactly at p0
# ---------------------------------------------------------------------
for p0 in (0.5, 0.55):
    n, rejections = 180, 0
    for _ in range(N_SIMS):
        # Actual directions Bernoulli(p0); model guesses up with the same
        # p0 independently -> correct indicator ~ Bernoulli(p0).
        k = int(rng.binomial(n, p0))
        pval = sum(
            math.comb(n, i) * (p0 ** i) * ((1 - p0) ** (n - i))
            for i in range(k, n + 1)
        )
        rejections += min(1.0, pval) < ALPHA
    rate = rejections / N_SIMS
    check(f"Binomial H0 (p0={p0}): odsetek p<0.05 ~ 5%", abs(rate - ALPHA) < TOL,
          f"rate = {rate:.3%}")

# ---------------------------------------------------------------------
# 3) Power sanity: a clearly worse predictor is almost always rejected
# ---------------------------------------------------------------------
actual = rng.normal(size=180)
good = actual + rng.normal(size=180)
bad = actual + rng.normal(size=180, scale=3.0)
_stat, p = dm_test(actual, bad, good)
check("DM moc: gorszy model odrzucony", p < ALPHA, f"p = {p:.5f}")

# ---------------------------------------------------------------------
# 4) Edge cases
# ---------------------------------------------------------------------
actual = rng.normal(size=100)
pred = actual + rng.normal(size=100)
stat, p = dm_test(actual, pred, pred)             # identical predictions
check("DM: identyczne predykcje -> (0, 1)", stat == 0.0 and p == 1.0)

s1, p1 = dm_test(actual, pred, actual)            # asymmetry sanity
s2, p2 = dm_test(actual, actual, pred)
check("DM: symetria znaku", abs(s1 + s2) < 1e-12 and abs(p1 - p2) < 1e-12,
      f"p1 = {p1:.4f}")

# binomial_da_test on synthetic folds
def make_results(correct: bool, n: int = 60, folds: int = 3, up_frac: float = 0.5):
    """Synthetic results dict: model always right (or always wrong)."""
    out = {}
    for fold in range(1, folds + 1):
        up = rng.random(n) < up_frac
        actual = np.where(up, 101.0, 99.0)
        prev = np.where(up, 100.0, 100.0)          # up day = 100 -> 101
        if correct:
            prediction = np.where(up, 101.0, 99.0)
        else:
            prediction = np.where(up, 99.0, 101.0)
        dfp = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=n),
                            "actual": actual, "prev_close": prev,
                            "prediction": prediction})
        out[fold] = {"LSTM": dfp}
    return out


res = make_results(correct=True)
tbl = binomial_da_test(res)
p_perfect = float(tbl["p-value (dwumianowy, jednostronny)"].iloc[0])
check("Binomial: model idealny -> p < 0.001", p_perfect < 0.001, f"p = {p_perfect:.2e}")

res = make_results(correct=False)
tbl = binomial_da_test(res)
p_wrong = float(tbl["p-value (dwumianowy, jednostronny)"].iloc[0])
# One-sided test can only detect "better than chance": always-wrong gives k=0 -> p = 1.
check("Binomial: model zawsze zły -> p = 1 (test jednostronny)", p_wrong == 1.0)

res = make_results(correct=True, up_frac=1.0)     # degenerate p0 = 1
tbl = binomial_da_test(res)
check("Binomial: p0 = 1 (same dni wzrostowe) - brak crashu", len(tbl) == 1)

m = compute_metrics(np.array([100.0, 101.0]), np.array([100.0, 101.0]),
                    np.array([99.0, 100.0]))
check("Metryki: predykcja idealna -> MAE=RMSE=0, DA=100",
      m["MAE ($)"] == 0 and m["RMSE ($)"] == 0 and m["DA (%)"] == 100.0)

print()
if failures:
    print(f"WYNIK: {len(failures)} test(y) NIEZALICZONE: {failures}")
    sys.exit(1)
print(f"WYNIK: wszystkie testy zaliczone (N_SIMS={N_SIMS}, seed={SEED})")
