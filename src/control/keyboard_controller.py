"""
Keyboard controller for MIT Mini Cheetah.

Key bindings:
  W/↑: Forward  S/↓: Backward  A/←: Strafe left  D/→: Strafe right
  Q: Turn left   E: Turn right   SPACE: Stop   SHIFT+dir: Run
  CTRL: Crouch   J: Jump   1/2/3: Walk/Trot/Run   X: Explore   ESC: Quit
"""
import threading
from typing import Optional

try:
    from pynput import keyboard as pynput_kb
    KEYBOARD_BACKEND = "pynput"
except ImportError:
    KEYBOARD_BACKEND = "none"


class KeyboardController:
    SPEED_WALK = 0.5
    SPEED_TROT = 1.5
    SPEED_RUN = 3.0
    SPEED_STRAFE = 0.8
    SPEED_TURN = 1.0

    def __init__(self):
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.mode = "trot"
        self.speed_level = self.SPEED_TROT
        self.crouching = False
        self.jumping = False
        self.exploring = False
        self.active = False
        self._keys_held = set()
        self._lock = threading.Lock()
        self._listener = None

    def start(self):
        if KEYBOARD_BACKEND == "pynput":
            self._listener = pynput_kb.Listener(
                on_press=self._on_press, on_release=self._on_release
            )
            self._listener.start()
            self.active = True
            print("[KeyboardController] Started (pynput)")
        else:
            print("[KeyboardController] No keyboard backend. Using scripted commands.")
            self.active = False

    def stop(self):
        if self._listener:
            self._listener.stop()
        self.active = False

    def get_command(self):
        """Returns (vx, vy, wz, mode_string)."""
        with self._lock:
            self._update_from_held_keys()
            return float(self.vx), float(self.vy), float(self.wz), self.mode

    def _update_from_held_keys(self):
        if self.exploring:
            return

        vx, vy, wz = 0.0, 0.0, 0.0
        speed = self.SPEED_RUN if "shift" in self._keys_held else self.speed_level
        if self.crouching:
            speed *= 0.3

        # Only one directional command is active — use the most recent key.
        # Priority order: last key pressed wins (pynput adds to _keys_held
        # in order, so we just pick the first match).
        keys = self._keys_held
        if "w" in keys or "up" in keys:
            vx = speed
        elif "s" in keys or "down" in keys:
            vx = -speed * 0.6
        elif "a" in keys or "left" in keys:
            vy = self.SPEED_STRAFE
        elif "d" in keys or "right" in keys:
            vy = -self.SPEED_STRAFE
        elif "q" in keys:
            wz = self.SPEED_TURN
        elif "e" in keys:
            wz = -self.SPEED_TURN

        self.vx, self.vy, self.wz = vx, vy, wz

    def _on_press(self, key):
        with self._lock:
            key_str = self._key_to_str(key)
            if key_str:
                self._keys_held.add(key_str)
            self._handle_mode_keys(key_str)

    def _on_release(self, key):
        with self._lock:
            key_str = self._key_to_str(key)
            self._keys_held.discard(key_str)
            if not any(k in self._keys_held for k in
                       ["w", "s", "a", "d", "up", "down", "left", "right", "q", "e"]):
                self.vx = self.vy = self.wz = 0.0

    def _handle_mode_keys(self, key_str):
        if key_str == "1":
            self.speed_level = self.SPEED_WALK; self.mode = "walk"
        elif key_str == "2":
            self.speed_level = self.SPEED_TROT; self.mode = "trot"
        elif key_str == "3":
            self.speed_level = self.SPEED_RUN; self.mode = "run"
        elif key_str == "j":
            self.jumping = True; self.mode = "jump"
        elif key_str == "ctrl" or key_str == "ctrl_l":
            self.crouching = not self.crouching
            self.mode = "crouch" if self.crouching else "trot"
        elif key_str == "space":
            self.vx = self.vy = self.wz = 0.0; self.mode = "stand"
        elif key_str == "x":
            self.exploring = not self.exploring
            self.mode = "explore" if self.exploring else "trot"

    @staticmethod
    def _key_to_str(key) -> Optional[str]:
        try:
            c = key.char
            return c.lower() if c else None
        except AttributeError:
            name = str(key).replace("Key.", "").lower()
            return name if name else None

    def print_bindings(self):
        print("""
╔══════════════════════════════════════════════════╗
║  MIT Mini Cheetah Keyboard Controls              ║
╠══════════════════════════════════════════════════╣
║  W / ↑        : Forward                         ║
║  S / ↓        : Backward                        ║
║  A / ←        : Strafe left                     ║
║  D / →        : Strafe right                    ║
║  Q            : Turn left                       ║
║  E            : Turn right                      ║
║  SHIFT + dir  : Run speed                       ║
║  CTRL         : Toggle crouch                   ║
║  J            : Jump                            ║
║  SPACE        : Stop / stand                    ║
║  1 / 2 / 3    : Walk / Trot / Run mode          ║
║  X            : Toggle exploration mode          ║
║  ESC          : Quit                            ║
╚══════════════════════════════════════════════════╝
""")
