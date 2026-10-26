import math

import pytest

from hermes_evals.stats import cohen_kappa, corrected_pass_rate, pass_rate, wilson_interval

Z95 = 1.959964


def test_all_passing_still_has_a_real_interval():
    # 5 cases, all pass: se is 0, so the interval uses n = 5 cases. Wilson's lower bound is n / (n + z^2).
    r = pass_rate({f"c{i}": [True] for i in range(5)})
    assert (r.rate, r.se, r.hi) == (1.0, 0.0, 1.0)
    assert r.lo == pytest.approx(5 / (5 + Z95**2), abs=1e-4)  # 0.566, not a collapsed 1.0


def test_case_means_set_the_standard_error():
    # Case scores 1, 1, 0, 0: p = 0.5, stdev = 0.577, se = 0.289. n_eff = 0.25 / 0.0833 = 3, raised to 4 cases.
    r = pass_rate({"a": [True, True], "b": [True, True], "c": [False, False], "d": [False, False]})
    assert r.rate == 0.5
    assert r.se == pytest.approx(0.288675, abs=1e-6)
    assert (r.lo, r.hi) == pytest.approx((0.1500, 0.8500), abs=1e-3)
    assert (r.n_cases, r.n_runs) == (4, 8)


def test_effective_sample_size_is_capped_at_the_number_of_runs():
    # Case scores barely vary, so the implied n_eff (63) exceeds the 16 runs; the interval uses 16.
    r = pass_rate({"a": [True, False] * 2, "b": [True, False] * 2, "c": [True, False] * 2,
                   "d": [True, True, True, False]})
    assert r.rate == pytest.approx(0.5625)
    assert (r.lo, r.hi) == pytest.approx(wilson_interval(0.5625, 16), abs=1e-6)


def test_cases_count_equally_whatever_their_repeats():
    r = pass_rate({"many": [True] * 10, "one": [False]})
    assert r.rate == 0.5 and (r.n_cases, r.n_runs) == (2, 11)


def test_a_single_case_has_no_standard_error_and_a_wide_interval():
    r = pass_rate({"only": [True, True, False]})
    assert math.isnan(r.se)
    assert (r.lo, r.hi) == pytest.approx(wilson_interval(2 / 3, 1), abs=1e-6)
    assert r.hi - r.lo > 0.7


def test_interval_stays_in_bounds_and_brackets_the_rate():
    r = pass_rate({"a": [False, False], "b": [False], "c": [True, False]})
    assert 0.0 <= r.lo <= r.rate <= r.hi <= 1.0


def test_no_runs_is_an_error():
    with pytest.raises(ValueError):
        pass_rate({"empty": []})


def test_kappa_hand_computed():
    # observed agreement 0.75, chance agreement 0.5 -> kappa 0.5
    assert cohen_kappa([True, True, False, False], [True, False, False, False]) == pytest.approx(0.5)


def test_kappa_exposes_a_judge_that_always_passes():
    truth = [True] * 9 + [False]
    assert cohen_kappa(truth, [True] * 10) == 0.0  # 90% raw agreement, zero skill


def test_corrected_pass_rate_undoes_a_lenient_judge():
    # True rate 0.6; judge TPR 0.9, TNR 0.7 -> observed = 0.6*0.9 + 0.4*0.3 = 0.66.
    assert corrected_pass_rate(0.66, 0.9, 0.7) == pytest.approx(0.6)
    assert math.isnan(corrected_pass_rate(0.5, 0.5, 0.5))
