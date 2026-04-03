"""
Terminal-based keyboard input for robot control.

Reads keystrokes directly from the terminal using raw mode (tty/termios).
Runs in a background thread so it works alongside the MuJoCo viewer
without requiring viewer window focus.

Key bindings (held-key style):
  W/↑: Forward   S/↓: Backward   A/←: Strafe left   D/→: Strafe right
  Q: Turn left    E: Turn right    SPACE: Stop all    1/2/3: Walk/Trot/Run
  J: Jump          C: Toggle crouch
  ESC: Quit
"""
import sys
import os
import threading
import time
import select
from typing import Tuple

# Speed presets
SPEEDS = {"walk": 0.5, "trot": 1.5, "run": 3.0}
STRAFE_SPEED = 0.8
TURN_SPEED = 1.0


class TerminalKeyController:
    """Non-blocking terminal keyboard controller using raw tty mode."""

    def __init__(self):
        self._fwd = False
        self._bwd = False
        self._left = False
        self._right = False
        self._turn_l = False
        self._turn_r = False
        self.speed = SPEEDS["trot"]
        self.mode = "stand"
        self.crouching = False
        self.jumping = False
        self._jump_counter = 0  # jump persists for N get_command reads
        self._pre_jump_mode = "stand"  # mode to restore when jump ends
        self.quit = False
        self._lock = threading.Lock()
        self._thread = None
        self._old_settings = None
        self._running = False

    def start(self):
        """Start the background key-reading thread."""
        if not os.isatty(sys.stdin.fileno()):
            print("[TerminalKeyController] stdin is not a tty, cannot read keys.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop reading and restore terminal settings."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _read_loop(self):
        """Background loop: put terminal in raw mode, read keypresses."""
        import tty
        import termios

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)  # cbreak mode: char-at-a-time, no echo
            while self._running:
                # Use select to avoid blocking forever
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    if not ch:
                        continue
                    # Handle escape sequences (arrow keys)
                    if ch == "\x1b":
                        # Could be ESC or start of escape sequence
                        if select.select([sys.stdin], [], [], 0.05)[0]:
                            ch2 = sys.stdin.read(1)
                            if ch2 == "[":
                                ch3 = sys.stdin.read(1)
                                self._handle_arrow(ch3)
                                continue
                        else:
                            # Bare ESC
                            with self._lock:
                                self.quit = True
                            return
                    self._handle_key(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _clear_directions(self):
        """Clear all directional toggles (called before setting a new one)."""
        self._fwd = self._bwd = False
        self._left = self._right = False
        self._turn_l = self._turn_r = False

    def _handle_key(self, ch: str):
        with self._lock:
            c = ch.lower()
            if c == "w":
                toggling_on = not self._fwd
                self._clear_directions()
                self._fwd = toggling_on
            elif c == "s":
                toggling_on = not self._bwd
                self._clear_directions()
                self._bwd = toggling_on
            elif c == "a":
                toggling_on = not self._left
                self._clear_directions()
                self._left = toggling_on
            elif c == "d":
                toggling_on = not self._right
                self._clear_directions()
                self._right = toggling_on
            elif c == "q":
                toggling_on = not self._turn_l
                self._clear_directions()
                self._turn_l = toggling_on
            elif c == "e":
                toggling_on = not self._turn_r
                self._clear_directions()
                self._turn_r = toggling_on
            elif c == " ":
                self._clear_directions()
                self.mode = "stand"
                self.crouching = False
                self.jumping = False
            elif c == "1":
                self.speed = SPEEDS["walk"]; self.mode = "walk"
            elif c == "2":
                self.speed = SPEEDS["trot"]; self.mode = "trot"
            elif c == "3":
                self.speed = SPEEDS["run"]; self.mode = "run"
            elif c == "j":
                self._pre_jump_mode = self.mode if self.mode != "jump" else self._pre_jump_mode
                self.jumping = True
                self._jump_counter = 75  # ~1.5s at 50Hz for full jump cycle
                self.crouching = False
                self.mode = "jump"
            elif c == "c":
                self.crouching = not self.crouching
                self.jumping = False
                self.mode = "crouch" if self.crouching else "trot"
            elif ch == "\x1b":  # ESC
                self.quit = True

    def _handle_arrow(self, ch3: str):
        with self._lock:
            if ch3 == "A":    # Up
                toggling_on = not self._fwd
                self._clear_directions()
                self._fwd = toggling_on
            elif ch3 == "B":  # Down
                toggling_on = not self._bwd
                self._clear_directions()
                self._bwd = toggling_on
            elif ch3 == "D":  # Left
                toggling_on = not self._left
                self._clear_directions()
                self._left = toggling_on
            elif ch3 == "C":  # Right
                toggling_on = not self._right
                self._clear_directions()
                self._right = toggling_on

    def get_command(self) -> Tuple[float, float, float, str]:
        with self._lock:
            vx = self.speed if self._fwd else (-self.speed * 0.6 if self._bwd else 0.0)
            vy = STRAFE_SPEED if self._left else (-STRAFE_SPEED if self._right else 0.0)
            wz = TURN_SPEED if self._turn_l else (-TURN_SPEED if self._turn_r else 0.0)
            mode = self.mode
            # Crouching reduces speed
            if self.crouching:
                vx *= 0.3
                vy *= 0.3
            # Jump persists for _jump_counter reads then reverts
            if self.jumping:
                self._jump_counter -= 1
                if self._jump_counter <= 0:
                    self.jumping = False
                    self.mode = self._pre_jump_mode
            if (vx != 0 or vy != 0 or wz != 0) and mode == "stand":
                mode = "trot"
            return vx, vy, wz, mode

    def reset_motion(self):
        """Reset all motion toggles (e.g. after episode reset)."""
        with self._lock:
            self._fwd = self._bwd = False
            self._left = self._right = False
            self._turn_l = self._turn_r = False
            self.mode = "stand"
            self.crouching = False
            self.jumping = False


def print_terminal_bindings():
    print("""
╔══════════════════════════════════════════════════╗
║  Unitree Go1 — Terminal Controls                ║
╠══════════════════════════════════════════════════╣
║  W / ↑        : Forward  (exclusive)            ║
║  S / ↓        : Backward (exclusive)            ║
║  A / ←        : Strafe left (exclusive)         ║
║  D / →        : Strafe right (exclusive)        ║
║  Q            : Turn left (exclusive)           ║
║  E            : Turn right (exclusive)          ║
║  J            : Jump                            ║
║  C            : Toggle crouch                   ║
║  1 / 2 / 3    : Walk / Trot / Run speed         ║
║  SPACE        : Stop all motion                 ║
║  ESC          : Quit                            ║
╠══════════════════════════════════════════════════╣
║  Input is from THIS TERMINAL — no need to       ║
║  click the MuJoCo viewer window.                ║
╚══════════════════════════════════════════════════╝
""")
