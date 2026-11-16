"""GARCH / GJR-GARCH filtered historical simulation VaR and ES research package.

Units used throughout: returns are percentage-point log returns (1% -> 1.0) and
losses are ``loss_pct = -return_pct`` for a long position.

Third-party reliance: GARCH/GJR parameter estimation and the one-step variance
forecast are delegated to the ``arch`` package; Ljung-Box statistics use
``statsmodels``; chi-square p-values use ``scipy``. Empirical VaR/ES,
historical and filtered historical simulation, the rolling forecast loop,
coverage-test likelihoods, metrics and reporting are implemented here.
"""

__version__ = "0.1.0"

MODEL_HS = "HS"
MODEL_GARCH = "GARCH-FHS"
MODEL_GJR = "GJR-FHS"
MODELS = (MODEL_HS, MODEL_GARCH, MODEL_GJR)
