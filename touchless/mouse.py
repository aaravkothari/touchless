"""OS cursor control. Thin wrapper so the backend is swappable later.

pyautogui's fail-safe is deliberately left ON: physically slam the mouse
into the top-left corner and the next cursor command raises
FailSafeException, which the app treats as "user pulled the emergency
brake". We inset our own movements from the screen edge so the app can't
trigger it by itself.
"""

import pyautogui

pyautogui.PAUSE = 0          # no artificial delay between commands
pyautogui.FAILSAFE = True    # corner slam = emergency stop


class Cursor:
    def __init__(self, inset_px: int):
        self.w, self.h = pyautogui.size()
        self.inset = inset_px

    def move_norm(self, nx: float, ny: float):
        """Move to normalized (0..1) screen coords, clamped inside the inset."""
        x = min(max(nx * self.w, self.inset), self.w - self.inset)
        y = min(max(ny * self.h, self.inset), self.h - self.inset)
        pyautogui.moveTo(x, y)

    def position(self) -> tuple[int, int]:
        p = pyautogui.position()
        return p.x, p.y

    def click(self):
        pyautogui.click()

    def down(self, button: str = "left"):
        pyautogui.mouseDown(button=button)

    def up(self, button: str = "left"):
        pyautogui.mouseUp(button=button)
