from __future__ import annotations

import numpy as np

from .math_utils import quaternion_to_R, rot_to_rpy_zxy, wrap_to_pi
from .model import QuadParams

class Controller:
    """Controller for quadrotor trajectory tracking."""

    def __init__(self, params: QuadParams) -> None:
        """Initialize controller with quadrotor parameters.

        Args:
            params: Quadrotor physical parameters (mass, inertia, etc.)
        """
        self.params = params

        # A slightly more conservative but more disturbance-robust tuning.
        self.Kp_pos = np.array([6.0, 6.0, 11.0])
        self.Kd_pos = np.array([4.5, 4.5, 6.0])
        self.Ki_pos = np.array([0.45, 0.45, 1.30])
        self.Kp_angle = np.array([10.5, 10.5, 4.0])
        self.Kd_angle = np.array([0.35, 0.35, 0.25])
        self.max_acc_xy = 8.2
        self.max_acc_z = 11.0
        self.int_pos_limit = np.array([0.8, 0.8, 0.6])
        self.prev_t = 0.0
        self.int_pos = np.zeros(3)

    def reset(self) -> None:
        """Reset controller state (if needed)."""
        self.prev_t = 0.0
        self.int_pos = np.zeros(3)

    def __call__(self, t: float, s: np.ndarray, s_des: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute control outputs (thrust and moments) for the quadrotor.

        Args:
            t: Current time [s]
            s: Current state vector (13,)
               [position, velocity, quaternion, angular_velocity]
            s_des: Desired state vector (11,)
               [position, velocity, acceleration, yaw, yaw_rate]

        Returns:
            F: Total thrust force [N]
            M: Moment vector [Mx, My, Mz] in body frame [N⋅m]
        """
        # Extract quadrotor parameters
        m = self.params.mass
        g = self.params.grav
        # ========================================================================
        # TODO: Implement your controller here
        # ========================================================================
        # 1. Unpack state and desired state
        pos = s[0:3]
        vel = s[3:6]
        q = s[6:10]
        omega = s[10:13]

        pos_des = s_des[0:3]
        vel_des = s_des[3:6]
        acc_des = s_des[6:9]
        yaw_des = s_des[9]
        yaw_rate_des = s_des[10]
        
        dt = float(np.clip(t - self.prev_t, 1e-3, 0.05)) if t > 0.0 else 0.01
        self.prev_t = t

        # Quaternion -> attitude
        R = quaternion_to_R(q)
        phi, theta, yaw = rot_to_rpy_zxy(R)

        # 2. Position loop
        e_pos = pos_des - pos
        e_vel = vel_des - vel
        self.int_pos += np.clip(e_pos, -0.5, 0.5) * dt
        self.int_pos = np.clip(self.int_pos, -self.int_pos_limit, self.int_pos_limit)

        a_cmd = acc_des + self.Kp_pos * e_pos + self.Kd_pos * e_vel + self.Ki_pos * self.int_pos
        a_cmd[0:2] = np.clip(a_cmd[0:2], -self.max_acc_xy, self.max_acc_xy)
        a_cmd[2] = float(np.clip(a_cmd[2], -self.max_acc_z, self.max_acc_z))

        # 3. Desired roll/pitch from lateral acceleration command
        phi_des = (a_cmd[0] * np.sin(yaw_des) - a_cmd[1] * np.cos(yaw_des)) / g
        theta_des = (a_cmd[0] * np.cos(yaw_des) + a_cmd[1] * np.sin(yaw_des)) / g
        phi_des = np.clip(phi_des, -self.params.maxangle, self.params.maxangle)
        theta_des = np.clip(theta_des, -self.params.maxangle, self.params.maxangle)

        # 4. Thrust with tilt compensation so altitude control remains effective
        tilt_comp = max(np.cos(phi) * np.cos(theta), 0.5)
        F = m * (g + a_cmd[2]) / tilt_comp
        F = float(np.clip(F, self.params.minF, self.params.maxF))

        # 5. Attitude loop
        e_angle = np.array(
            [
                wrap_to_pi(phi_des - phi),
                wrap_to_pi(theta_des - theta),
                wrap_to_pi(yaw_des - yaw),
            ]
        )
        e_omega = np.array([0.0, 0.0, yaw_rate_des]) - omega
        M = self.Kp_angle * e_angle + self.Kd_angle * e_omega
        # ========================================================================
        # End of your implementation
        # ========================================================================

        return F, M
