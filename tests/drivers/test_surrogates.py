import numpy as np

from NDC.drivers.oscillators import SineRhythmDriver
from NDC.drivers.surrogates import PhaseRandomizedDriver


def test_phase_randomized_preserves_psd():
    base = SineRhythmDriver(amplitude=1.0, frequency=2.0, phase=0.1)
    rng = np.random.default_rng(42)
    surrogate = PhaseRandomizedDriver.from_driver(base, 0.0, 2.0, 0.01, rng)

    times = np.arange(0.0, 2.0 + 0.01 * 0.5, 0.01)
    base_trace = np.array([base(t).rhythm for t in times])
    surrogate_trace = np.array([surrogate(t).rhythm for t in times])

    base_psd = np.abs(np.fft.rfft(base_trace)) ** 2
    surrogate_psd = np.abs(np.fft.rfft(surrogate_trace)) ** 2

    np.testing.assert_allclose(base_psd, surrogate_psd, rtol=1e-6, atol=1e-8)
