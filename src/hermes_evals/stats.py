"""Pass rates with standard errors and confidence intervals, clustered by case.

Method (following Miller, "Adding Error Bars to Evals", Anthropic 2024):

1. Average each case's repeats into one case score, then average case scores. Cases are the sampled unit;
   repeats only reduce noise within a case, so a case with more repeats doesn't count more.
2. The standard error is the spread of case scores over sqrt(number of cases): clustered by case, so
   correlated repeats can't shrink it.
3. The interval is a Wilson score interval (Wilson, 1927) at the effective sample size implied by that
   standard error. Unlike mean ± z·SE it stays inside [0, 1] and keeps a real width when every run passes
   or every run fails.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist, fmean, stdev


@dataclass(frozen=True, slots=True)
class PassRate:
    """A pass rate with its uncertainty, for a whole run or one category of cases.

    `rate` weights every case equally, however many times it was repeated. `se`, `lo` and `hi` treat
    cases, not individual runs, as the independent samples, so repeats can't make the result look more
    certain than the number of cases supports.
    """

    rate: float  # mean of per-case pass rates
    se: float  # standard error, clustered by case; nan with a single case
    lo: float  # lower bound of the confidence interval
    hi: float  # upper bound of the confidence interval
    n_cases: int
    n_runs: int


def pass_rate(runs_by_case: Mapping[str, Sequence[bool]], confidence: float = 0.95) -> PassRate:
    """Summarize pass/fail results, given as each case's list of run outcomes, into a PassRate.

    Raises ValueError if there are no runs. See the module docstring for the method.
    """
    case_rates = [fmean(runs) for runs in runs_by_case.values() if runs]
    if not case_rates:
        raise ValueError("no runs to summarize")
    n_cases = len(case_rates)
    n_runs = sum(len(runs) for runs in runs_by_case.values())
    p = fmean(case_rates)
    se = stdev(case_rates) / math.sqrt(n_cases) if n_cases > 1 else math.nan

    # Effective sample size: how many independent pass/fail draws would give this standard error. It lies
    # between n_cases (repeats fully agree) and n_runs (repeats independent); when the standard error is
    # zero or undefined, use n_cases, the conservative end.
    n_eff = p * (1 - p) / se**2 if se > 0 else n_cases
    n_eff = min(max(n_eff, n_cases), n_runs)

    lo, hi = wilson_interval(p, n_eff, confidence)
    return PassRate(rate=p, se=se, lo=lo, hi=hi, n_cases=n_cases, n_runs=n_runs)


def wilson_interval(p: float, n: float, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion p observed over n trials.

    Instead of centering on p (as p ± z·sqrt(p(1-p)/n) does), it pulls the center toward 0.5 and widens
    the interval in proportion to how few trials there are. That keeps both bounds inside [0, 1] and gives
    a real width even at p = 0 or p = 1, where the plain formula collapses to a point. The clamp only
    guards against floating-point error.
    """
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, center - half), min(1.0, center + half)


def cohen_kappa(truth: Sequence[bool], predicted: Sequence[bool]) -> float:
    """Agreement between two raters beyond what chance alone would produce: 1 is perfect, 0 is chance level.

    Kept as a secondary number. Judge-versus-human agreement is reported mainly as TPR and TNR, which show
    which way a judge errs; kappa folds both into one number and is best suited to comparing two humans.
    """
    n = len(truth)
    if n == 0:
        return math.nan
    observed = sum(t == p for t, p in zip(truth, predicted, strict=True)) / n
    p_truth, p_pred = sum(truth) / n, sum(predicted) / n
    expected = p_truth * p_pred + (1 - p_truth) * (1 - p_pred)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def corrected_pass_rate(observed: float, tpr: float, tnr: float) -> float:
    """Estimate the true pass rate from a judge's observed pass rate, given the judge's TPR and TNR.

    A judge that passes some bad responses and fails some good ones skews the observed rate. Inverting
    observed = true·TPR + (1 - true)·(1 - TNR) gives true = (observed + TNR - 1) / (TPR + TNR - 1)
    (Rogan & Gladen, 1978). Undefined when TPR + TNR <= 1, a judge no better than chance.
    """
    if tpr + tnr <= 1:
        return math.nan
    return min(1.0, max(0.0, (observed + tnr - 1) / (tpr + tnr - 1)))
