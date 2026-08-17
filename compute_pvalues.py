#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical comparison of U-net, V-net and E-SegNet
Thalamic segmentation study — Pediatric Research, PR-2025-2288

Reproduces every p value reported in Section II.10 (Statistical analysis),
Section III.6 (Results) and the Discussion.

TEST
----
Exact two-sided Wilcoxon signed-rank test for paired samples. Each test
volume is segmented by all three architectures, so observations are paired
volume by volume. A non-parametric test is used because normality cannot be
verified at this sample size.

CAVEAT REPORTED IN THE MANUSCRIPT
---------------------------------
The test ranks the paired differences. With n pairs there are only 2**n
equally likely sign patterns under H0, so the smallest two-sided p value the
test can ever return is 2 / 2**n:

    n = 4  ->  p_min = 0.125
    n = 5  ->  p_min = 0.0625
    n = 6  ->  p_min = 0.03125    <- first n that can reach p < 0.05

With four paired volumes NO comparison can reach the conventional 0.05
threshold, however large the observed difference. The tests are therefore
reported as descriptive and exploratory, not as evidence of superiority.

DATA PROVENANCE  (important)
----------------------------
The per-volume values below are stated explicitly rather than parsed, so
that the analysis is auditable and reproducible independently of the layout
of metrics.txt. They were taken from metrics.txt as follows:

  U-net     second block (4-volume run, includes the Volume column)
  V-net     FIRST block  — the corrected V-net table (MAD 0.31/0.29/0.24/0.37)
  E-SegNet  second block (4-volume run)

The V-net table in the second block of metrics.txt is a STALE copy from
before the MAD correction (it still carries MAD 0.57/0.53/0.43/0.68, mean
0.55, which contradicts the 0.30 reported in Table 3). It must not be used.
Run this script with `--check path/to/metrics.txt` to re-verify the values
below against the file once it has been cleaned up.

Volumes: the four test volumes segmented by all three architectures under
identical conditions. Patient49 is absent from the corrected V-net run and
is therefore excluded, since the Wilcoxon test requires complete pairs.

Requirements: numpy, scipy       ->  pip install numpy scipy
Usage:        python compute_pvalues.py
              python compute_pvalues.py --check metrics.txt
"""

import itertools
import sys

import numpy as np
from scipy import stats

MODELS = ["U-net", "V-net", "E-SegNet"]
VOLUMES = ["Patient37", "Patient48", "Patient53", "Patient81"]

# metric -> {model: per-volume values, in the order of VOLUMES}
# DSC: higher is better.  ASV, MSD, RVD: lower is better.
DATA = {
    "DSC": {
        "U-net":    [0.88, 0.87, 0.91, 0.85],
        "V-net":    [0.89, 0.89, 0.92, 0.87],
        "E-SegNet": [0.91, 0.89, 0.92, 0.88],
    },
    "ASV": {                                   # Z_Grad column in metrics.txt
        "U-net":    [0.6529, 0.6563, 0.6439, 0.6244],
        "V-net":    [0.5691, 0.6003, 0.6269, 0.5672],
        "E-SegNet": [0.5708, 0.6185, 0.6251, 0.5620],
    },
    "MSD": {                                   # MAD column, mm
        "U-net":    [0.32, 0.32, 0.25, 0.43],
        "V-net":    [0.31, 0.29, 0.24, 0.37],  # corrected values, first block
        "E-SegNet": [0.25, 0.29, 0.23, 0.34],
    },
    "RVD": {                                   # relative volume difference, %
        "U-net":    [8.58, 16.47, 0.88, 14.63],
        "V-net":    [13.13, 12.37, 6.64, 8.96],
        "E-SegNet": [8.63, 11.96, 5.25, 12.01],
    },
}


def describe():
    """Mean +/- SD per model and per metric (SD with ddof=1, as in Table 3)."""
    print(f"Descriptive statistics  (n = {len(VOLUMES)}: {', '.join(VOLUMES)})\n")
    print(f"{'metric':8s}{'model':11s}{'mean':>10s}{'SD':>9s}")
    for metric, series in DATA.items():
        for model in MODELS:
            x = np.array(series[model], dtype=float)
            print(f"{metric:8s}{model:11s}{x.mean():10.4f}{x.std(ddof=1):9.4f}")
        print()


def wilcoxon_table():
    """All pairwise exact two-sided Wilcoxon signed-rank tests."""
    n = len(VOLUMES)
    print(f"Exact two-sided Wilcoxon signed-rank test  (n = {n} paired volumes)")
    print(f"Smallest p value attainable at n = {n}:  2 / 2**{n} = {2 / 2 ** n:.5f}\n")
    print(f"{'metric':8s}{'comparison':26s}{'mean diff':>11s}{'ties':>6s}{'W':>6s}{'p':>9s}")
    for metric, series in DATA.items():
        for a, b in itertools.combinations(MODELS, 2):
            x = np.array(series[a], dtype=float)
            y = np.array(series[b], dtype=float)
            d = x - y
            # zero_method='wilcox' discards tied pairs before ranking
            W, p = stats.wilcoxon(x, y, zero_method="wilcox",
                                  alternative="two-sided", method="exact")
            print(f"{metric:8s}{a + ' vs ' + b:26s}{d.mean():+11.4f}"
                  f"{int((d == 0).sum()):6d}{W:6.1f}{p:9.4f}")
        print()


def rank_detail(metric="ASV", a="V-net", b="E-SegNet"):
    """Signed-rank breakdown of one comparison.

    Explains to the reader why V-net vs E-SegNet gives p = 1.00 for ASV:
    the two rank sums are exactly equal, i.e. the observed split is precisely
    what the null hypothesis predicts.
    """
    x = np.array(DATA[metric][a], dtype=float)
    y = np.array(DATA[metric][b], dtype=float)
    d = x - y
    ranks = stats.rankdata(np.abs(d))
    print(f"Signed-rank detail — {metric}, {a} vs {b}\n")
    print(f"{'volume':11s}{a:>10s}{b:>10s}{'diff':>10s}{'rank':>6s}{'lower':>11s}")
    for v, xi, yi, di, ri in zip(VOLUMES, x, y, d, ranks):
        print(f"{v:11s}{xi:10.4f}{yi:10.4f}{di:+10.4f}{ri:6.0f}"
              f"{(a if di < 0 else b):>11s}")
    w_b, w_a = ranks[d > 0].sum(), ranks[d < 0].sum()
    print(f"\n  sum of ranks where {b} is lower : {w_b:.0f}")
    print(f"  sum of ranks where {a} is lower : {w_a:.0f}")
    W, p = stats.wilcoxon(x, y, zero_method="wilcox",
                          alternative="two-sided", method="exact")
    print(f"  W = min = {W:.0f}   ->   p = {p:.4f}")
    if w_a == w_b:
        print("\n  The two rank sums are identical: the models are ranked in "
              "opposite\n  directions in equal measure, which is exactly what H0 "
              "predicts. Hence p = 1.00.")


def null_distribution(n=4):
    """Exact null distribution of W, to show where p_min comes from."""
    counts = {}
    for signs in itertools.product([0, 1], repeat=n):
        W = sum(r for s, r in zip(signs, range(1, n + 1)) if s)
        counts[W] = counts.get(W, 0) + 1
    total = 2 ** n
    print(f"\nExact null distribution of W for n = {n} ({total} equally likely "
          f"sign patterns)\n")
    for W in sorted(counts):
        print(f"  W = {W:2d} : {counts[W]:2d}/{total}"
              f"{'   <- most extreme' if W in (0, n * (n + 1) // 2) else ''}")
    print(f"\n  Two-sided p for the most extreme outcome = 2 x 1/{total} = "
          f"{2 / total:.5f}\n  No result can fall below this value.")


def check_against_file(path):
    """Re-read metrics.txt and report any value that differs from DATA."""
    import re
    col_of = {"DSC": "Dice", "ASV": "Z_Grad", "MSD": "MAD", "RVD": "ΔVr %"}
    canon = {"u-net": "U-net", "v-net": "V-net", "e-segnet": "E-SegNet"}
    found, model, cols = {}, None, None
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        m = re.match(r"^(U-?net|V-?net|E-?Segnet)\s*$", s, re.I)
        if m:
            model = canon[m.group(1).lower()]
            continue
        if s.startswith("Patient ") and "|" in s:
            cols = [c.strip() for c in s.split("|")][1:]
            continue
        m = re.match(r"^(Patient\d+)\S*\s*\|(.+)$", s)
        if m and model and cols:
            vals = dict(zip(cols, [float(v) for v in m.group(2).split("|")]))
            found.setdefault((model, m.group(1)), []).append(vals)

    print(f"Cross-check against {path}\n")
    problems = 0
    for metric, series in DATA.items():
        col = col_of[metric]
        for model in MODELS:
            for v, expected in zip(VOLUMES, series[model]):
                rows = found.get((model, v), [])
                seen = {r[col] for r in rows if col in r}
                if not seen:
                    print(f"  MISSING   {metric:4s} {model:9s} {v}")
                    problems += 1
                elif expected not in seen:
                    print(f"  DIFFERS   {metric:4s} {model:9s} {v}: "
                          f"script {expected}, file {sorted(seen)}")
                    problems += 1
                elif len(seen) > 1:
                    print(f"  AMBIGUOUS {metric:4s} {model:9s} {v}: file contains "
                          f"{sorted(seen)}; script uses {expected}")
                    problems += 1
    print("\n  All values match the file." if problems == 0
          else f"\n  {problems} discrepancy/ies — see DATA PROVENANCE in the docstring.")


def main():
    if "--check" in sys.argv:
        check_against_file(sys.argv[sys.argv.index("--check") + 1])
        return
    describe()
    wilcoxon_table()
    rank_detail()
    null_distribution(len(VOLUMES))


if __name__ == "__main__":
    main()
