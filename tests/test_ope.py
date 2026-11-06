import numpy as np

from ridepay import ope


def test_shift_density_integrates_to_one():
    r = 0.15
    x = np.linspace(np.log1p(-r), np.log1p(r), 200_001)
    assert abs(np.trapezoid(ope.shift_density(x, r), x) - 1) < 1e-3


def test_kernel_ipw_recovers_a_known_curve():
    rng = np.random.default_rng(0)
    n, r, h = 400_000, 0.15, 0.02
    s = rng.uniform(-r, r, n)
    x = np.log1p(s)
    p = 1 / (1 + np.exp(-(0.2 + 4 * x)))                       # true acceptance at shift x
    y = (rng.random(n) < p).astype(float)
    target = np.full(n, 0.07)
    est, w = ope.kernel_ipw(y, x, target, np.full(n, -1.0), r, h)
    truth = 1 / (1 + np.exp(-(0.2 + 4 * 0.07)))
    assert abs(est - truth) < 0.01
    assert (w > 0).sum() < n                                    # only trips near the target count


def test_targets_outside_the_test_range_or_on_the_floor_are_unsupported():
    r, h = 0.15, 0.02
    floor_shift = np.array([-1.0, -1.0, 0.0, -1.0])
    x_target = np.array([0.05, 0.20, 0.0, np.log1p(-0.149)])
    assert ope.supported(x_target, floor_shift, r, h).tolist() == [True, False, False, False]
