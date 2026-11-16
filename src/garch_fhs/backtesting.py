"""VaR coverage tests: Kupiec (1995) and Christoffersen (1998).

All log-likelihoods are evaluated with ``scipy.special.xlogy`` so that
``0 * log(0) = 0``. p-values come from ``scipy.stats.chi2``.

Kupiec unconditional coverage (``n`` observations, ``x`` exceptions,
nominal probability ``p``, ``pi = x / n``)::

    LR_uc = -2 [ (n-x) log(1-p) + x log p - (n-x) log(1-pi) - x log pi ]  ~ chi2(1)

Christoffersen independence, using transition counts ``n_ij`` (state i at
``t-1`` -> state j at ``t``) over the ``n-1`` consecutive pairs::

    pi01 = n01/(n00+n01),  pi11 = n11/(n10+n11),  pi1 = (n01+n11)/(n-1)
    L1   = n00 log(1-pi01) + n01 log pi01 + n10 log(1-pi11) + n11 log pi11
    L0   = (n00+n10) log(1-pi1) + (n01+n11) log pi1
    LR_ind = -2 (L0 - L1)                                                   ~ chi2(1)

Conditional coverage is evaluated on the same ``n-1`` transitions::

    Lp    = (n00+n10) log(1-p) + (n01+n11) log p
    LR_cc = -2 (Lp - L1) = LR_uc(on the n-1 transition outcomes) + LR_ind   ~ chi2(2)

so the decomposition holds exactly. Tests are reported as ``undefined``
(statistic and p-value NaN, reason stated) when a transition row has no
observations - e.g. no exceptions, exceptions only on the last day, or no
non-exception days - because the alternative likelihood is then not
identified. Asymptotic chi-square p-values ignore parameter-estimation
error from rolling windows and may be unreliable in small samples.
"""

from __future__ import annotations

import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2


def _result(test: str, statistic: float, df: int, n: int, x: int, status: str, note: str) -> dict:
    p_value = float(chi2.sf(statistic, df)) if status == "defined" else float("nan")
    return {
        "test": test,
        "statistic": float(statistic) if status == "defined" else float("nan"),
        "df": df,
        "p_value": p_value,
        "n": n,
        "exceptions": x,
        "status": status,
        "note": note,
    }


def _bernoulli_loglik(n0: int, n1: int, prob: float) -> float:
    return float(xlogy(n0, 1.0 - prob) + xlogy(n1, prob))


def _low_power_note(n: int, p: float) -> str:
    return "expected exceptions < 5: very low power" if n * p < 5 else ""


def kupiec_test(exceptions: np.ndarray, p: float) -> dict:
    """Kupiec proportion-of-failures likelihood-ratio test."""
    e = np.asarray(exceptions, dtype=int)
    n, x = int(e.size), int(e.sum())
    if n == 0:
        return _result("kupiec_uc", np.nan, 1, 0, 0, "undefined", "no observations")
    pi = x / n
    lr = -2.0 * (_bernoulli_loglik(n - x, x, p) - _bernoulli_loglik(n - x, x, pi))
    return _result("kupiec_uc", max(lr, 0.0), 1, n, x, "defined", _low_power_note(n, p))


def transition_counts(exceptions: np.ndarray) -> dict[str, int]:
    """Counts ``n00, n01, n10, n11`` of consecutive exception-state pairs."""
    e = np.asarray(exceptions, dtype=int)
    prev, cur = e[:-1], e[1:]
    return {
        "n00": int(np.sum((prev == 0) & (cur == 0))),
        "n01": int(np.sum((prev == 0) & (cur == 1))),
        "n10": int(np.sum((prev == 1) & (cur == 0))),
        "n11": int(np.sum((prev == 1) & (cur == 1))),
    }


def _degenerate_reason(c: dict[str, int]) -> str:
    if c["n00"] + c["n01"] == 0:
        return "no transitions from the non-exception state"
    if c["n10"] + c["n11"] == 0:
        return "no transitions from the exception state (no exceptions before the final day)"
    return ""


def christoffersen_tests(exceptions: np.ndarray, p: float) -> list[dict]:
    """Christoffersen independence and conditional-coverage tests."""
    e = np.asarray(exceptions, dtype=int)
    n, x = int(e.size), int(e.sum())
    if n < 2:
        return [
            _result("christoffersen_ind", np.nan, 1, n, x, "undefined", "fewer than two observations"),
            _result("christoffersen_cc", np.nan, 2, n, x, "undefined", "fewer than two observations"),
        ]
    c = transition_counts(e)
    reason = _degenerate_reason(c)
    counts_note = "n00={n00}, n01={n01}, n10={n10}, n11={n11}".format(**c)
    if reason:
        note = f"{reason}; {counts_note}"
        return [
            _result("christoffersen_ind", np.nan, 1, n, x, "undefined", note),
            _result("christoffersen_cc", np.nan, 2, n, x, "undefined", note),
        ]
    pi01 = c["n01"] / (c["n00"] + c["n01"])
    pi11 = c["n11"] / (c["n10"] + c["n11"])
    pi1 = (c["n01"] + c["n11"]) / (n - 1)
    l1 = _bernoulli_loglik(c["n00"], c["n01"], pi01) + _bernoulli_loglik(c["n10"], c["n11"], pi11)
    l0 = _bernoulli_loglik(c["n00"] + c["n10"], c["n01"] + c["n11"], pi1)
    lp = _bernoulli_loglik(c["n00"] + c["n10"], c["n01"] + c["n11"], p)
    lr_ind = max(-2.0 * (l0 - l1), 0.0)
    lr_cc = max(-2.0 * (lp - l1), 0.0)
    extra = _low_power_note(n, p)
    note = counts_note + (f"; {extra}" if extra else "")
    if c["n11"] == 0:
        note += "; no consecutive exceptions observed"
    return [
        _result("christoffersen_ind", lr_ind, 1, n, x, "defined", note),
        _result("christoffersen_cc", lr_cc, 2, n, x, "defined", note),
    ]


def coverage_tests(exceptions: np.ndarray, p: float) -> list[dict]:
    """Kupiec UC, Christoffersen IND and CC for one exception sequence."""
    return [kupiec_test(exceptions, p), *christoffersen_tests(exceptions, p)]
