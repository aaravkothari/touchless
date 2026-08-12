"""Click generation: dwell (hover to click) and deliberate-blink."""

import time

from .config import Config


class DwellClicker:
    """Click when the cursor stays inside a small circle for dwell_time_s."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.anchor: tuple[float, float] | None = None
        self.anchor_t = 0.0
        self.last_click_t = 0.0

    def update(self, x: float, y: float) -> bool:
        now = time.monotonic()
        if now - self.last_click_t < self.cfg.dwell_cooldown_s:
            self.anchor = None
            return False
        if self.anchor is None:
            self.anchor = (x, y)
            self.anchor_t = now
            return False
        dx, dy = x - self.anchor[0], y - self.anchor[1]
        if dx * dx + dy * dy > self.cfg.dwell_radius_px ** 2:
            self.anchor = (x, y)
            self.anchor_t = now
            return False
        if now - self.anchor_t >= self.cfg.dwell_time_s:
            self.last_click_t = now
            self.anchor = None
            return True
        return False

    def progress(self) -> float:
        """0..1 fraction of the dwell timer, for UI feedback."""
        if self.anchor is None:
            return 0.0
        return min((time.monotonic() - self.anchor_t) / self.cfg.dwell_time_s, 1.0)


class PinchClicker:
    """Click when thumb and index tip pinch together (hand mode).

    Fires once per pinch: requires release above the threshold before the
    next click, plus a cooldown.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.was_pinched = False
        self.last_click_t = 0.0

    def update(self, pinch: float) -> bool:
        now = time.monotonic()
        pinched = pinch < self.cfg.pinch_click_threshold
        fire = (pinched and not self.was_pinched
                and now - self.last_click_t >= self.cfg.pinch_cooldown_s)
        self.was_pinched = pinched
        if fire:
            self.last_click_t = now
        return fire


class BlinkClicker:
    """Click on a deliberate blink: eyes closed for blink_min_s..blink_max_s.

    Reflex blinks (~0.1-0.15 s) fall under blink_min_s and are ignored.
    Input is the model's eyeBlink blendshape score (0 open .. 1 closed).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.closed_since: float | None = None
        self.last_click_t = 0.0

    def update(self, blink: float) -> bool:
        now = time.monotonic()
        closed = blink > self.cfg.blink_closed_threshold

        if closed:
            if self.closed_since is None:
                self.closed_since = now
            return False

        # Eyes just opened: was it a deliberate blink?
        if self.closed_since is not None:
            duration = now - self.closed_since
            self.closed_since = None
            if (self.cfg.blink_min_s <= duration <= self.cfg.blink_max_s
                    and now - self.last_click_t >= self.cfg.blink_cooldown_s):
                self.last_click_t = now
                return True
        return False
