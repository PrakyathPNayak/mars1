"""
Exploration controller: given a target heading, generate locomotion commands.

Uses proportional heading control:
  - Compute heading error
  - Turn proportionally until aligned
  - Move forward when aligned
"""
import math
import numpy as np


class ExplorationPolicy:
    def __init__(
        self,
        turn_gain: float = 1.5,
        forward_speed: float = 1.5,
        align_threshold: float = 0.15,
        arrival_radius: float = 0.3,
    ):
        self.turn_gain = turn_gain
        self.forward_speed = forward_speed
        self.align_threshold = align_threshold
        self.arrival_radius = arrival_radius
        self.target_heading = 0.0
        self.target_pos = None
        self.arrived = False

    def set_target_heading(self, heading_rad: float):
        self.target_heading = heading_rad
        self.target_pos = None
        self.arrived = False

    def set_target_waypoint(self, target_x: float, target_y: float,
                             current_x: float, current_y: float):
        dx = target_x - current_x
        dy = target_y - current_y
        self.target_heading = math.atan2(dy, dx)
        self.target_pos = (target_x, target_y)
        self.arrived = False

    def get_command(self, current_yaw: float, current_pos: tuple = (0.0, 0.0)) -> tuple:
        """Returns (vx, vy, wz) locomotion command."""
        if self.arrived:
            return 0.0, 0.0, 0.0

        if self.target_pos is not None:
            dx = self.target_pos[0] - current_pos[0]
            dy = self.target_pos[1] - current_pos[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < self.arrival_radius:
                self.arrived = True
                return 0.0, 0.0, 0.0
            self.target_heading = math.atan2(dy, dx)

        heading_err = self._wrap_angle(self.target_heading - current_yaw)
        wz = float(np.clip(self.turn_gain * heading_err, -2.0, 2.0))
        align_factor = math.exp(-4.0 * heading_err ** 2)
        vx = float(np.clip(self.forward_speed * align_factor, 0.0, self.forward_speed))

        return vx, 0.0, wz

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def get_status(self) -> dict:
        return {
            "target_heading_deg": math.degrees(self.target_heading),
            "target_pos": self.target_pos,
            "arrived": self.arrived,
        }
