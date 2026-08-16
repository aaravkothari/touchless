"""One Euro filter — the standard choice for cursor-like signals.

Plain moving averages force you to pick between lag and jitter. The One Euro
filter adapts: heavy smoothing when the signal is slow (kills jitter while
you're holding still), light smoothing when it moves fast (low lag when you
flick your gaze). Reference: Casiez et al., CHI 2012.
"""

import math


class _LowPass:
    def __init__(self):
        self.prev: float | None = None

    def apply(self, x: float, alpha: float) -> float:
        if self.prev is None:
            self.prev = x
        self.prev = alpha * x + (1.0 - alpha) * self.prev
        return self.prev


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._t: float | None = None

    def apply(self, x: float, t: float) -> float:
        if self._t is None:
            self._t = t
            self._x.apply(x, 1.0)
            self._dx.apply(0.0, 1.0)
            return x
        dt = max(t - self._t, 1e-6)
        self._t = t
        dx = (x - (self._x.prev if self._x.prev is not None else x)) / dt
        edx = self._dx.apply(dx, _alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x.apply(x, _alpha(cutoff, dt))

    def reset(self):
        self._x = _LowPass()
        self._dx = _LowPass()
        self._t = None


class OneEuro2D:
    """Convenience wrapper: one filter per axis, shared parameters.

    d_cutoff low-passes the velocity estimate that drives the beta term.
    Lowering it below the 1 Hz default matters for jittery signals: raw
    frame-to-frame noise has huge instantaneous velocity, so with a fast
    d_cutoff the beta boost fires ON the noise - smoothing is reduced
    exactly when the hand is still and needs it most.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.fx = OneEuro(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.fy = OneEuro(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

    def apply(self, x: float, y: float, t: float) -> tuple[float, float]:
        return self.fx.apply(x, t), self.fy.apply(y, t)

    def reset(self):
        self.fx.reset()
        self.fy.reset()
