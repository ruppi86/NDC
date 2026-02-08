import numpy as np

from ndc_analysis.mnps import compute_mnps_with_diagnostics, reconstruct_from_mnps


def test_mnps_roundtrip_identity():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(50, 4))
    result, diag = compute_mnps_with_diagnostics(Y, k=4, normalize="zscore")
    Y_hat = reconstruct_from_mnps(result.X, diag)
    assert np.allclose(Y_hat, Y, atol=1e-6)
