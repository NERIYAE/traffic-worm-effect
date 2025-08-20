#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
traffic_platoon_sim.py

Traffic "worm/accordion" effect demo at a green light with 4 vehicles.
Generates three GIFs:
  - baseline_idm.gif          : human reaction delay (IDM-based) → worm effect
  - coordinated_cacc.gif      : centrally coordinated platoon (CACC)
  - stacked_comparison_en.gif : vertical comparison (top=CACC, bottom=worm)

Dependencies: numpy, matplotlib, pillow
Usage:
  python traffic_platoon_sim.py

Model notes
-----------
Baseline: IDM (Intelligent Driver Model) with explicit reaction delay 'tau' on
followers, causing stop-and-go wave propagation.

Coordinated: simple Cooperative Adaptive Cruise Control (CACC) with spacing
policy d_des = s0 + T_h * v, PD terms on spacing/relative speed, and feed-forward
of the leader's acceleration for string stability.

This code is educational. It is not a traffic engineering tool.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")  # offscreen rendering for saving GIFs
import matplotlib.pyplot as plt
from matplotlib import animation, patches

# -----------------------
# Global simulation params
# -----------------------
DT: float = 0.1                 # [s] integration step
T_SIM: float = 40.0             # [s] total time
STEPS: int = int(T_SIM / DT)

N_CARS: int = 4                 # number of vehicles
L: float = 4.5                  # vehicle length [m]
ROAD_LEN: float = 220.0         # render window [m]
STOP_LINE_X: float = 0.0        # stop line at x=0

V_DES: float = 13.9             # desired speed ~ 50 km/h [m/s]
A_MAX: float = 2.0              # accel clamp [m/s^2]
B_MAX: float = 3.0              # decel clamp [m/s^2]

# Initial queue behind stop line
BASE_GAP: float = 2.0           # standstill bumper gap [m]
X0: np.ndarray = np.array([-(L + BASE_GAP) * i - 6.0 for i in range(N_CARS)], dtype=float)
V_INIT: np.ndarray = np.zeros(N_CARS, dtype=float)


# -----------------------------
# Baseline: IDM + reaction delay
# -----------------------------
def simulate_baseline_idm(
    x_init: np.ndarray,
    v_init: np.ndarray,
    tau: float = 1.1,         # [s] human reaction delay
    s0: float = 2.0,          # [m] jam distance
    T: float = 1.6,           # [s] desired headway
    a: float = 1.2,           # [m/s^2] IDM accel parameter
    b: float = 2.0,           # [m/s^2] IDM braking parameter
    v_des: float = V_DES
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    IDM with explicit reaction delay for followers.
    Leader starts accelerating after 'tau'. Followers react with state delay.

    Returns:
        x [STEPS,N_CARS]: positions [m]
        v [STEPS,N_CARS]: speeds   [m/s]
        a_hist [STEPS,N_CARS]: accelerations [m/s^2]
    """
    x = np.zeros((STEPS, N_CARS))
    v = np.zeros((STEPS, N_CARS))
    a_hist = np.zeros((STEPS, N_CARS))
    x[0] = x_init.copy()
    v[0] = v_init.copy()

    delay_steps = max(1, int(tau / DT))

    for t in range(1, STEPS):
        for i in range(N_CARS):
            if i == 0:
                # Leader: green at t=0, reacts after 'tau'
                acc = 0.0 if t < delay_steps else a * (1.0 - (v[t - 1, i] / v_des) ** 4)
            else:
                # Follower reacts to DELAYED lead state
                t_ref = max(0, t - delay_steps)
                dx = (x[t - 1, i - 1] - x[t - 1, i]) - L
                dv = v[t - 1, i] - v[t_ref, i - 1]
                s_star = s0 + v[t - 1, i] * T + (v[t - 1, i] * dv) / (2.0 * np.sqrt(a * b) + 1e-6)
                dx_safe = max(0.1, dx)
                acc = a * (1.0 - (v[t - 1, i] / v_des) ** 4 - (s_star / dx_safe) ** 2)

            # Clamp accel
            acc = float(np.clip(acc, -B_MAX, A_MAX))
            a_hist[t, i] = acc

        # Integrate kinematics
        v[t] = np.maximum(0.0, v[t - 1] + a_hist[t] * DT)
        x[t] = x[t - 1] + v[t] * DT

    return x, v, a_hist


# -------------------------
# Coordinated: simple CACC
# -------------------------
def simulate_coordinated_cacc(
    x_init: np.ndarray,
    v_init: np.ndarray,
    s0: float = 2.0,          # [m] standstill spacing
    T_h: float = 0.6,         # [s] time headway
    kp: float = 0.9,          # spacing gain
    kv: float = 1.1,          # relative speed gain
    a_ff: float = 0.6,        # feed-forward of leader accel
    v_des: float = V_DES
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple CACC controller:
      acc_i = kp*(gap - (s0 + T_h*v_i)) + kv*(v_{i-1} - v_i) + a_ff*acc_{i-1}
    Leader tracks v_des with a P term.
    """
    x = np.zeros((STEPS, N_CARS))
    v = np.zeros((STEPS, N_CARS))
    a_hist = np.zeros((STEPS, N_CARS))
    x[0] = x_init.copy()
    v[0] = v_init.copy()

    for t in range(1, STEPS):
        # Leader: proportional control to v_des
        e_v0 = v_des - v[t - 1, 0]
        a_hist[t, 0] = float(np.clip(0.8 * e_v0, -B_MAX, A_MAX))

        # Followers: PD on spacing/speed + feed-forward
        for i in range(1, N_CARS):
            dx = (x[t - 1, i - 1] - x[t - 1, i]) - L
            dv = v[t - 1, i - 1] - v[t - 1, i]
            d_des = s0 + T_h * v[t - 1, i]
            e_gap = dx - d_des
            acc = kp * e_gap + kv * dv + a_ff * a_hist[t, i - 1]
            a_hist[t, i] = float(np.clip(acc, -B_MAX, A_MAX))

        v[t] = np.maximum(0.0, v[t - 1] + a_hist[t] * DT)
        x[t] = x[t - 1] + v[t] * DT

    return x, v, a_hist


# ----------------
# Animation helpers
# ----------------
def make_animation_positions(x_traj: np.ndarray, title: str, outfile: str) -> None:
    """Animate rectangles for each car along the lane."""
    fig, ax = plt.subplots(figsize=(7, 2.1))
    ax.set_xlim(-60, ROAD_LEN)
    ax.set_ylim(-1, 3)
    ax.set_xlabel("x [m]")
    ax.set_yticks([])
    ax.set_title(title)
    ax.axvline(STOP_LINE_X, linestyle="--")

    cars: list[patches.Rectangle] = []
    for i in range(N_CARS):
        rect = patches.Rectangle((x_traj[0, i] - L / 2, 0.5), L, 1.0, fill=False)
        ax.add_patch(rect)
        cars.append(rect)

    time_text = ax.text(0.01, 0.95, "", transform=ax.transAxes, ha="left", va="top")

    def init():
        for i, r in enumerate(cars):
            r.set_x(x_traj[0, i] - L / 2)
        time_text.set_text("t = 0.0 s")
        return cars + [time_text]

    def update(frame):
        for i, r in enumerate(cars):
            r.set_x(x_traj[frame, i] - L / 2)
        time_text.set_text(f"t = {frame * DT:.1f} s")
        return cars + [time_text]

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=STEPS, interval=30, blit=True)
    anim.save(outfile, writer="pillow", fps=int(1.0 / DT))
    plt.close(fig)


def make_stacked_comparison(x_base: np.ndarray, x_coord: np.ndarray, outfile: str) -> None:
    """
    Single figure with two horizontal bands:
      top band   → coordinated (CACC)
      bottom band→ baseline worm effect
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.set_xlim(-60, ROAD_LEN)
    ax.set_ylim(0, 6)
    ax.set_xlabel("x [m]")
    ax.set_yticks([])
    ax.set_title("1) Without worm effect (CACC)  |  2) With worm effect")
    ax.axhline(2.8, linestyle="--")     # visual separator between bands
    ax.axvline(STOP_LINE_X, linestyle="--")

    # Bands' vertical placement
    band_bottom_y = 0.6   # baseline at bottom band
    band_top_y    = 3.4   # CACC at top band
    h = 0.9

    cars_baseline: list[patches.Rectangle] = []
    cars_cacc: list[patches.Rectangle] = []

    # Bottom band: baseline
    for i in range(N_CARS):
        r1 = patches.Rectangle((x_base[0, i] - L / 2, band_bottom_y), L, h, fill=False)
        ax.add_patch(r1)
        cars_baseline.append(r1)

    # Top band: CACC
    for i in range(N_CARS):
        r2 = patches.Rectangle((x_coord[0, i] - L / 2, band_top_y), L, h, fill=False)
        ax.add_patch(r2)
        cars_cacc.append(r2)

    # Labels inside axes
    ax.text(0.01, 0.90, "1) Without worm effect (CACC)", transform=ax.transAxes, ha="left", va="top")
    ax.text(0.01, 0.43, "2) With worm effect",           transform=ax.transAxes, ha="left", va="top")
    time_text = ax.text(0.99, 0.05, "", transform=ax.transAxes, ha="right", va="bottom")

    def init():
        for i in range(N_CARS):
            cars_baseline[i].set_x(x_base[0, i] - L / 2)
            cars_cacc[i].set_x(x_coord[0, i] - L / 2)
        time_text.set_text("t = 0.0 s")
        return cars_baseline + cars_cacc + [time_text]

    def update(frame):
        for i in range(N_CARS):
            cars_baseline[i].set_x(x_base[frame, i] - L / 2)
            cars_cacc[i].set_x(x_coord[frame, i] - L / 2)
        time_text.set_text(f"t = {frame * DT:.1f} s")
        return cars_baseline + cars_cacc + [time_text]

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=STEPS, interval=30, blit=True)
    anim.save(outfile, writer="pillow", fps=int(1.0 / DT))
    plt.close(fig)


# -------------
# Main pipeline
# -------------
def main() -> None:
    # Baseline and coordinated trajectories
    x_baseline, _, _ = simulate_baseline_idm(X0, V_INIT, tau=1.1)
    x_cacc, _, _ = simulate_coordinated_cacc(X0, V_INIT)

    # Individual animations
    make_animation_positions(x_baseline, "Baseline: worm effect (reaction delay)", "baseline_idm.gif")
    make_animation_positions(x_cacc,     "Coordinated: central CACC platoon",     "coordinated_cacc.gif")

    # Stacked comparison
    make_stacked_comparison(x_baseline, x_cacc, "stacked_comparison_en.gif")

    print("Saved: baseline_idm.gif, coordinated_cacc.gif, stacked_comparison_en.gif")


if __name__ == "__main__":
    main()
