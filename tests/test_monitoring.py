import numpy as np
import pandas as pd

import config
from src.monitoring import psi, psi_report


def test_psi_zero_for_identical():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) < 1e-6


def test_psi_positive_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(2, 1, 5000)          # shifted by 2 std
    assert psi(a, b) > 0.2


def test_psi_report_flags_drift():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({f: rng.lognormal(9, 1, 3000) for f in config.DRIFT_FEATURES})
    surged = pd.DataFrame({f: rng.lognormal(9, 1, 3000) * 3 for f in config.DRIFT_FEATURES})
    assert psi_report(ref, surged)["drift"] is True
    assert psi_report(ref, ref)["drift"] is False
