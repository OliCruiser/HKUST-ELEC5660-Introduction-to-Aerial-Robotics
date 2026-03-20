from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def _poly_derivative(coeffs: np.ndarray) -> np.ndarray:
    """Derivative of a polynomial with coefficients in descending powers."""
    degree = len(coeffs) - 1
    if degree <= 0:
        return np.array([0.0])
    return np.array([coeffs[i] * (degree - i) for i in range(degree)], dtype=float)


def _solve_equality_qp(H: np.ndarray, Aeq: np.ndarray, beq: np.ndarray) -> np.ndarray:
    """Solve min 0.5 x^T H x s.t. Aeq x = beq using KKT system."""
    H = 0.5 * (H + H.T)
    n = H.shape[0]
    m = Aeq.shape[0]
    KKT = np.zeros((n + m, n + m))
    KKT[:n, :n] = H
    KKT[:n, n:] = Aeq.T
    KKT[n:, :n] = Aeq
    rhs = np.zeros(n + m)
    rhs[n:] = beq
    sol, _, _, _ = np.linalg.lstsq(KKT, rhs, rcond=None)
    return sol[:n]


def _generate_smooth_only(waypoints: np.ndarray, n_seg: int, total_time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate smooth trajectory using quintic polynomials.
    Ensures position, velocity, and acceleration continuity at waypoints.
    Uses 5th order polynomials (6 coefficients per segment).
    """
    # 这部分我们采用一个“足够平滑、但不做最优”的朴素做法：
    #
    # 1. 先给每一段分配时间 T_scale
    # 2. 再给每个航点分配一个“经过该点时的速度 v_i 和加速度 a_i”
    # 3. 对每一段 [p_i -> p_{i+1}]，构造一个五次多项式：
    #       p(t) = c0 + c1 t + c2 t^2 + c3 t^3 + c4 t^4 + c5 t^5
    #    让它同时满足：
    #       起点位置 / 速度 / 加速度
    #       终点位置 / 速度 / 加速度
    #
    # 因为相邻两段共享同一个 waypoint 的 v_i 和 a_i，
    # 所以天然就保证了 C^2 连续：位置、速度、加速度连续。

    # 为了让每段飞行时间和路径长度匹配，我们按段长比例分配 total_time。
    seg_vec = waypoints[1:] - waypoints[:-1] # 相邻航点的向量差，shape (n_seg-1, 3)
    seg_len = np.linalg.norm(seg_vec, axis=1) # 每段的长度，shape (n_seg-1,)
    seg_len = np.maximum(seg_len, 1e-3)
    T_scale = total_time * seg_len / np.sum(seg_len)

    n_wp = waypoints.shape[0]

    # v_wp[i] / a_wp[i] 分别表示通过第 i 个航点时的速度和加速度。
    # 端点通常设为 0，表示从静止起飞、最终静止停下。
    v_wp = np.zeros((n_wp, 3))
    a_wp = np.zeros((n_wp, 3))

    # 中间点的速度用“左右相邻段斜率的平均”近似。
    # 这样做的直觉是：不要在 waypoint 处突然拐得太死。
    for i in range(1, n_wp - 1):
        slope_prev = (waypoints[i] - waypoints[i - 1]) / T_scale[i - 1]
        slope_next = (waypoints[i + 1] - waypoints[i]) / T_scale[i]
        v_wp[i] = 0.5 * (slope_prev + slope_next)

    # 中间点的加速度用速度差分近似。
    # 这是一个简单但实用的方式，用来让轨迹不仅速度连续，也更平滑。
    for i in range(1, n_wp - 1):
        dt = 0.5 * (T_scale[i - 1] + T_scale[i])
        a_wp[i] = (v_wp[i + 1] - v_wp[i - 1]) / max(2.0 * dt, 1e-3)

    # 系数矩阵按“从低次到高次”存储：
    # [c0, c1, c2, c3, c4, c5]
    # 这样和后面 evaluate 里 [::-1] + polyval 的用法正好对应。
    c_x = np.zeros((6, n_seg))
    c_y = np.zeros((6, n_seg))
    c_z = np.zeros((6, n_seg))

    def solve_quintic(p0: float, v0: float, a0: float, p1: float, v1: float, a1: float, T: float) -> np.ndarray:
        """Solve one 1D quintic polynomial segment."""
        # 已知：
        # p(0), p'(0), p''(0), p(T), p'(T), p''(T)
        # 求 6 个系数 c0~c5
        A = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                [1.0, T, T**2, T**3, T**4, T**5],
                [0.0, 1.0, 2.0 * T, 3.0 * T**2, 4.0 * T**3, 5.0 * T**4],
                [0.0, 0.0, 2.0, 6.0 * T, 12.0 * T**2, 20.0 * T**3],
            ],
            dtype=float,
        )
        b = np.array([p0, v0, a0, p1, v1, a1], dtype=float)
        return np.linalg.solve(A, b)

    # 对每一段、每一个坐标轴分别求 quintic 系数。
    for seg in range(n_seg):
        T = T_scale[seg]

        coeff_x = solve_quintic(
            waypoints[seg, 0], v_wp[seg, 0], a_wp[seg, 0],
            waypoints[seg + 1, 0], v_wp[seg + 1, 0], a_wp[seg + 1, 0],
            T,
        )
        coeff_y = solve_quintic(
            waypoints[seg, 1], v_wp[seg, 1], a_wp[seg, 1],
            waypoints[seg + 1, 1], v_wp[seg + 1, 1], a_wp[seg + 1, 1],
            T,
        )
        coeff_z = solve_quintic(
            waypoints[seg, 2], v_wp[seg, 2], a_wp[seg, 2],
            waypoints[seg + 1, 2], v_wp[seg + 1, 2], a_wp[seg + 1, 2],
            T,
        )

        c_x[:, seg] = coeff_x
        c_y[:, seg] = coeff_y
        c_z[:, seg] = coeff_z

    return c_x, c_y, c_z, T_scale


def _generate_minimum_jerk(waypoints: np.ndarray, n_seg: int, total_time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate minimum jerk trajectory using 6th order polynomials.
    Minimizes the integral of jerk (third derivative) squared.
    """
    # ========================================================================
    # TODO: Implement your minimum jerk trajectory generation here
    # ========================================================================


    # ========================================================================
    # End of your implementation
    # ========================================================================
    raise NotImplementedError


def _generate_minimum_snap(waypoints: np.ndarray, n_seg: int, total_time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate minimum snap trajectory using 8th order polynomials.
    Minimizes the integral of snap (fourth derivative) squared.
    """
    # ========================================================================
    # TODO: Implement your minimum snap trajectory generation here
    # ========================================================================


    # ========================================================================
    # End of your implementation
    # ========================================================================
    raise NotImplementedError


@dataclass
class TrajectoryGenerator:
    """Polynomial trajectory generator with multiple methods."""

    waypoints: np.ndarray
    method: str = "snap"
    total_time: float = 25.0

    def __post_init__(self) -> None:
        self.waypoints = np.asarray(self.waypoints, dtype=float)
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 3:
            raise ValueError("waypoints must be shaped (N, 3)")
        self.n_seg = self.waypoints.shape[0] - 1
        if self.n_seg < 1:
            raise ValueError("need at least two waypoints")

        if self.method not in {"smooth", "jerk", "snap"}:
            raise ValueError("method must be 'smooth', 'jerk', or 'snap'")

        # Set polynomial order based on method
        if self.method == "smooth":
            self.n = 6  # quintic (5th order, 6 coefficients)
        elif self.method == "jerk":
            self.n = 6  # 6th order
        else:  # snap
            self.n = 8  # 8th order

        self._prepare()

    def _prepare(self) -> None:
        """Prepare trajectory by calling appropriate generation method."""
        if self.method == "smooth":
            self.c_x, self.c_y, self.c_z, self.T_scale = _generate_smooth_only(
                self.waypoints, self.n_seg, self.total_time
            )
        elif self.method == "jerk":
            self.c_x, self.c_y, self.c_z, self.T_scale = _generate_minimum_jerk(
                self.waypoints, self.n_seg, self.total_time
            )
        else:  # snap
            self.c_x, self.c_y, self.c_z, self.T_scale = _generate_minimum_snap(
                self.waypoints, self.n_seg, self.total_time
            )

    def _segment_time(self, t: float) -> tuple[int, float]:
        t = float(t)
        for seg in range(self.n_seg):
            if t - self.T_scale[seg] <= 0.0:
                return seg, t
            t -= self.T_scale[seg]
        # Clamp to the end of the last segment
        return self.n_seg - 1, self.T_scale[self.n_seg - 1]

    def evaluate(self, t: float) -> np.ndarray:
        seg, local_t = self._segment_time(t)
        coeff_x = self.c_x[:, seg][::-1]
        coeff_y = self.c_y[:, seg][::-1]
        coeff_z = self.c_z[:, seg][::-1]

        s_des = np.zeros(11)
        s_des[0] = np.polyval(coeff_x, local_t)
        s_des[1] = np.polyval(coeff_y, local_t)
        s_des[2] = np.polyval(coeff_z, local_t)

        dcoeff_x = _poly_derivative(coeff_x)
        dcoeff_y = _poly_derivative(coeff_y)
        dcoeff_z = _poly_derivative(coeff_z)

        s_des[3] = np.polyval(dcoeff_x, local_t)
        s_des[4] = np.polyval(dcoeff_y, local_t)
        s_des[5] = np.polyval(dcoeff_z, local_t)

        return s_des
