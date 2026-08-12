"""Learned gaze model: candidates compete on held-out data, best one wins.

Pipeline: pursuit data -> hygiene filtering -> pursuit-lag compensation
(searched, not assumed) -> temporal train/holdout split -> train three
candidate models -> pick the lowest holdout error -> refit on everything.

Candidates:
  ridge  - ridge regression on the 17-term physics expansion (the old
           closed-form approach; kept as the baseline that keeps us honest)
  mlp    - small neural net (2x64, standardized inputs, early stopping)
  hgb    - gradient-boosted trees, one per screen axis

Every calibration prints the comparison table, so each run doubles as a
model experiment on your own face.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config
from .pursuit import PursuitData

MODEL_VERSION = 4

# Feature indices (see tracking.FEATURE_NAMES).
_GX, _GY, _YAW, _PITCH, _ROLL, _TX, _TY, _TZ = range(8)


# ---------------------------------------------------------------- candidates

def _expand(F: np.ndarray) -> np.ndarray:
    """(n, 8) raw features -> (n, 17) physics terms (bias, linear, tz*angles, angles^2)."""
    ang = F[:, [_GX, _GY, _YAW, _PITCH]]
    return np.hstack([np.ones((len(F), 1)), F, F[:, [_TZ]] * ang, ang * ang])


class RidgePhysics:
    """The previous closed-form model, as a competing candidate."""

    def __init__(self, lambdas: tuple[float, ...]):
        self.lambdas = lambdas
        self.lam: float | None = None
        self.col_mean = self.col_std = self.W = None

    def _design(self, X: np.ndarray) -> np.ndarray:
        Z = (_expand(X) - self.col_mean) / self.col_std
        return np.clip(Z, -4.0, 4.0)

    def _solve(self, D: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
        P = np.eye(D.shape[1])
        P[0, 0] = 0.0
        return np.linalg.solve(D.T @ D + lam * P, D.T @ Y)

    def fit(self, X, Y, Xva=None, Yva=None):
        E = _expand(X)
        self.col_mean = E.mean(axis=0)
        self.col_std = E.std(axis=0) + 1e-9
        self.col_mean[0], self.col_std[0] = 0.0, 1.0
        D = self._design(X)
        if self.lam is None:
            if Xva is not None and len(Xva):
                errs = [(float(np.mean((self._design(Xva) @ self._solve(D, Y, lam) - Yva) ** 2)), lam)
                        for lam in self.lambdas]
                self.lam = min(errs)[1]
            else:
                self.lam = self.lambdas[1]
        self.W = self._solve(D, Y, self.lam)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._design(X) @ self.W


class AxisHGB:
    """Gradient-boosted trees, one regressor per screen axis."""

    def __init__(self):
        self.rx = HistGradientBoostingRegressor(max_iter=300, early_stopping=True,
                                                random_state=0)
        self.ry = HistGradientBoostingRegressor(max_iter=300, early_stopping=True,
                                                random_state=0)

    def fit(self, X, Y, Xva=None, Yva=None):
        self.rx.fit(X, Y[:, 0])
        self.ry.fit(X, Y[:, 1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([self.rx.predict(X), self.ry.predict(X)])


class SkMLP:
    """Small neural net on standardized features."""

    def __init__(self):
        self.pipe = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500,
                         early_stopping=True, n_iter_no_change=10,
                         random_state=0),
        )

    def fit(self, X, Y, Xva=None, Yva=None):
        self.pipe.fit(X, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipe.predict(X)


def _candidates(cfg: Config) -> dict:
    return {"ridge": RidgePhysics(cfg.ridge_lambdas),
            "mlp": SkMLP(),
            "hgb": AxisHGB()}


# ------------------------------------------------------------ dataset build

def build_dataset(data: PursuitData, cfg: Config, lag_ms: float):
    """Apply hygiene filtering + pursuit-lag label shift.

    The eyes trail a moving target, so the feature at time t is paired with
    where the target was `lag` seconds earlier. Returns (X, Y, t, phase).
    """
    lag = lag_ms / 1000.0
    Xs, Ys, ts, ps = [], [], [], []
    for ph in np.unique(data.phase):
        m = data.phase == ph
        t, X, blink, tg = data.t[m], data.X[m], data.blink[m], data.target[m]

        # Drop blinks and a short tail after each one (gaze rebounds late).
        ok = blink < cfg.blink_gate
        bad_t = t[~ok]
        if len(bad_t):
            recovering = np.any((t[:, None] > bad_t[None, :])
                                & (t[:, None] - bad_t[None, :] < cfg.blink_after_s), axis=1)
            ok &= ~recovering

        # Label shift: target position `lag` seconds before each sample.
        Y = np.column_stack([np.interp(t - lag, t, tg[:, 0]),
                             np.interp(t - lag, t, tg[:, 1])])
        Xs.append(X[ok]); Ys.append(Y[ok]); ts.append(t[ok])
        ps.append(np.full(int(ok.sum()), ph))
    return (np.concatenate(Xs), np.concatenate(Ys),
            np.concatenate(ts), np.concatenate(ps))


def _temporal_split(t, phase, frac):
    """Boolean holdout mask: the last `frac` of each phase, by time.

    Random splits would leak — adjacent frames are near-duplicates.
    """
    val = np.zeros(len(t), dtype=bool)
    for ph in np.unique(phase):
        m = phase == ph
        cutoff = np.quantile(t[m], 1.0 - frac)
        val |= m & (t > cutoff)
    return val


# ------------------------------------------------------------------- model

@dataclass
class GazeModel:
    model: object          # fitted candidate with .predict((n,8)) -> (n,2)
    lag_ms: float
    name: str
    holdout_px: float      # winner's holdout error at fit time (px, mean)

    def predict(self, f: np.ndarray) -> tuple[float, float]:
        out = self.model.predict(f[None, :])[0]
        return float(out[0]), float(out[1])

    def save(self, path: str):
        joblib.dump({"version": MODEL_VERSION, "model": self.model,
                     "lag_ms": self.lag_ms, "name": self.name,
                     "holdout_px": self.holdout_px}, path)

    @classmethod
    def load(cls, path: str) -> "GazeModel":
        d = joblib.load(path)
        if d.get("version") != MODEL_VERSION:
            raise ValueError(
                f"{path} is from an incompatible version - recalibrate:\n"
                "  python -m touchless calibrate"
            )
        return cls(model=d["model"], lag_ms=d["lag_ms"], name=d["name"],
                   holdout_px=d["holdout_px"])

    @classmethod
    def fit(cls, data: PursuitData, cfg: Config,
            screen_w: int, screen_h: int) -> "GazeModel":
        px_scale = np.array([screen_w, screen_h])

        def val_px(model, Xva, Yva) -> float:
            d = (model.predict(Xva) - Yva) * px_scale
            return float(np.mean(np.hypot(d[:, 0], d[:, 1])))

        # 1. Lag search with the cheap ridge baseline (lag is a property of
        #    eyes-vs-target timing, not of the downstream model).
        best_lag, best_err = cfg.lag_grid_ms[0], np.inf
        for lag in cfg.lag_grid_ms:
            X, Y, t, ph = build_dataset(data, cfg, lag)
            val = _temporal_split(t, ph, cfg.holdout_frac)
            r = RidgePhysics(cfg.ridge_lambdas).fit(X[~val], Y[~val], X[val], Y[val])
            err = val_px(r, X[val], Y[val])
            if err < best_err:
                best_lag, best_err = lag, err
        print(f"pursuit lag: {best_lag} ms (ridge holdout {best_err:.0f}px)")

        # 2. Candidates compete at the winning lag.
        X, Y, t, ph = build_dataset(data, cfg, best_lag)
        val = _temporal_split(t, ph, cfg.holdout_frac)
        print(f"dataset: {len(X)} samples ({int(val.sum())} held out)")
        results = {}
        for name, cand in _candidates(cfg).items():
            t0 = time.monotonic()
            cand.fit(X[~val], Y[~val], X[val], Y[val])
            results[name] = (val_px(cand, X[val], Y[val]), cand,
                             time.monotonic() - t0)
        print("model comparison (holdout, lower is better):")
        for name, (err, _, dt) in sorted(results.items(), key=lambda kv: kv[1][0]):
            print(f"  {name:6s} {err:6.0f}px   (fit {dt:.1f}s)")
        win_name, (win_err, _, _) = min(results.items(), key=lambda kv: kv[1][0])

        # 3. Refit the winner on ALL data before shipping it.
        final = _candidates(cfg)[win_name]
        if isinstance(final, RidgePhysics):
            final.lam = results[win_name][1].lam  # keep the selected lambda
        final.fit(X, Y, X[val], Y[val])
        print(f"winner: {win_name} ({win_err:.0f}px holdout), refit on all data")
        return cls(model=final, lag_ms=float(best_lag), name=win_name,
                   holdout_px=win_err)
