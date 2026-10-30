"""
rotary_actuator_sim.py
======================

Software-first simulation of a single-turn rotary actuator position-control
system:

    commanded angle -> wraparound-aware error -> discrete PID controller
    -> PWM / H-bridge (averaged) -> brushed DC motor + gearbox + load
    -> output-shaft absolute encoder (quantized) -> back to controller

IMPORTANT: every numerical parameter below is an ILLUSTRATIVE PLACEHOLDER chosen
to give a physically plausible small geared-DC-motor actuator. None of them is a
datasheet value for a real part. Replace them with measured / datasheet values
before drawing hardware conclusions.

Units inside the plant: SI (rad, rad/s, N*m, V, A, s).
Units at the user / controller interface: degrees.

Run:  python rotary_actuator_sim.py      (writes figures + results.json to ./out)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, replace, asdict
from typing import List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEG = math.pi / 180.0
RAD = 180.0 / math.pi


# =============================================================================
# 1. Angle utilities (the wraparound logic lives here)
# =============================================================================
def wrap_360(a_deg: float) -> float:
    """Map any angle to [0, 360)."""
    return a_deg % 360.0


def wrap_pm180(e_deg: float) -> float:
    """Map any angle difference to (-180, 180]."""
    e = e_deg % 360.0          # [0, 360)
    if e > 180.0:
        e -= 360.0             # (-180, 180]
    return e


def shortest_move_deg(target_deg: float, current_deg: float,
                      tie_band_deg: float = 0.5) -> float:
    """
    Signed shortest angular move from current to target, in degrees.

    * Result lies in (-180, 180] for ordinary cases.
    * Tie rule: if the move is within `tie_band_deg` of +/-180 deg (both
      directions essentially equal), we ALWAYS choose the positive (forward)
      direction. Without a band, encoder quantization/noise could make the
      sign flip between e.g. +179.99 and -179.99 depending on one LSB.
    """
    e = wrap_pm180(target_deg - current_deg)
    if e < -(180.0 - tie_band_deg):   # e.g. -179.8 -> treat as tie, go forward
        e += 360.0                    # becomes +180.2
    return e


# =============================================================================
# 2. Parameter containers (change these independently)
# =============================================================================
@dataclass
class MotorParams:
    R: float = 2.0        # armature resistance [ohm]          (placeholder)
    L: float = 1.0e-3     # armature inductance [H]            (placeholder)
    Kt: float = 0.02      # torque constant [N*m/A]            (placeholder)
    Ke: float = 0.02      # back-EMF constant [V*s/rad] (=Kt in SI)
    J_m: float = 1.0e-6   # rotor inertia [kg*m^2]             (placeholder)
    b_m: float = 1.0e-6   # rotor viscous friction [N*m*s/rad] (placeholder)


@dataclass
class MechParams:
    N: float = 50.0             # gear ratio (motor turns per output turn)
    J_load: float = 1.0e-4      # load inertia at output [kg*m^2]
    b_load: float = 1.0e-4      # load viscous friction at output [N*m*s/rad]
    tau_coulomb: float = 0.02   # Coulomb friction at output [N*m]
    omega_s: float = 0.01       # tanh smoothing speed for friction [rad/s]
    # NOT modelled: gear backlash, compliance, stiction > Coulomb, efficiency.

    def J_eq(self, m: MotorParams) -> float:
        """Total inertia reflected to the output shaft."""
        return self.J_load + self.N ** 2 * m.J_m

    def b_eq(self, m: MotorParams) -> float:
        return self.b_load + self.N ** 2 * m.b_m


@dataclass
class DriveParams:
    V_supply: float = 12.0   # H-bridge supply [V]   (placeholder, NOT verified)
    pwm_bits: int = 8        # duty-cycle resolution (8-bit timer)
    # The H-bridge is modelled as an ideal averaged voltage source: V = duty*Vs.
    # NOT modelled: switching ripple, dead time, driver voltage drop, current limit.


@dataclass
class EncoderParams:
    bits: int = 14               # 14-bit absolute single-turn (16384 counts/rev)
    noise_lsb_rms: float = 0.0   # additive Gaussian noise, in LSB
    zero_offset_deg: float = 0.0 # mounting offset (calibrated out in firmware)


@dataclass
class ControllerParams:
    Kp: float = 40.0          # [V/rad]
    Ki: float = 400.0         # [V/(rad*s)]  (set 0 for pure PD)
    Kd: float = 0.4           # [V*s/rad]
    Ts: float = 1.0e-3        # control period [s] (1 kHz)
    vel_filter_hz: float = 150.0  # LPF cutoff on differentiated encoder angle
    i_zone_deg: float = 5.0   # integrate only when |error| < this (integral separation)
    i_limit_V: float = 3.0    # clamp on integral contribution [V]
    tie_band_deg: float = 0.5 # see shortest_move_deg
    delay_samples: int = 1    # computation/actuation delay, in control periods


@dataclass
class SimConfig:
    dt: float = 1.0e-4        # plant integration step [s] (RK4)
    t_end: float = 8.0        # [s]
    theta0_deg: float = 0.0   # initial true output angle [deg]
    omega0: float = 0.0       # initial output speed [rad/s]
    seed: int = 0


# targets: list of (time_s, angle_deg in [0,360))
# disturbances: list of (t_on, t_off, torque_Nm at output, + = positive direction)
Target = Tuple[float, float]
Disturbance = Tuple[float, float, float]


# =============================================================================
# 3. Plant model: armature circuit + rigid geared mechanism
# =============================================================================
class Plant:
    """
    States (output-referred):
        theta [rad]  - output-shaft angle (continuous, NOT wrapped)
        omega [rad/s]- output-shaft speed
        i     [A]    - armature current

    Equations:
        L di/dt      = V - R i - Ke * N * omega
        J_eq dw/dt   = N Kt i - b_eq omega - tau_c * tanh(omega/omega_s) + tau_d
        dtheta/dt    = omega
    """

    def __init__(self, m: MotorParams, g: MechParams):
        self.m, self.g = m, g
        self.J = g.J_eq(m)
        self.b = g.b_eq(m)

    def deriv(self, x, V, tau_d):
        th, w, i = x
        m, g = self.m, self.g
        di = (V - m.R * i - m.Ke * g.N * w) / m.L
        tau_f = g.tau_coulomb * math.tanh(w / g.omega_s)
        dw = (g.N * m.Kt * i - self.b * w - tau_f + tau_d) / self.J
        return (w, dw, di)

    def rk4(self, x, V, tau_d, dt):
        # V and tau_d are held constant across the step (ZOH from controller,
        # disturbance sampled at the start of the step).
        k1 = self.deriv(x, V, tau_d)
        x2 = tuple(xi + 0.5 * dt * ki for xi, ki in zip(x, k1))
        k2 = self.deriv(x2, V, tau_d)
        x3 = tuple(xi + 0.5 * dt * ki for xi, ki in zip(x, k2))
        k3 = self.deriv(x3, V, tau_d)
        x4 = tuple(xi + dt * ki for xi, ki in zip(x, k3))
        k4 = self.deriv(x4, V, tau_d)
        return tuple(xi + dt / 6.0 * (a + 2 * b + 2 * c + d)
                     for xi, a, b, c, d in zip(x, k1, k2, k3, k4))


# =============================================================================
# 4. Encoder model
# =============================================================================
class Encoder:
    """Absolute single-turn encoder on the OUTPUT shaft: returns [0,360) deg,
    quantized to 2^bits counts, with optional Gaussian noise."""

    def __init__(self, p: EncoderParams, rng: np.random.Generator):
        self.p, self.rng = p, rng
        self.cpr = 2 ** p.bits

    def read_deg(self, theta_rad: float) -> float:
        a = wrap_360(theta_rad * RAD + self.p.zero_offset_deg)
        counts = a / 360.0 * self.cpr
        if self.p.noise_lsb_rms > 0:
            counts += self.rng.normal(0.0, self.p.noise_lsb_rms)
        counts = int(math.floor(counts)) % self.cpr
        # firmware removes the (calibrated) mounting offset
        return wrap_360(counts * 360.0 / self.cpr - self.p.zero_offset_deg)


# =============================================================================
# 5. Controller (what would run on the microcontroller)
# =============================================================================
class PositionController:
    """
    Discrete PID position controller with:
      * software unwrapping of the single-turn encoder into a continuous angle
      * shortest-path target latching (direction decided ONCE per command)
      * derivative on measurement (no derivative kick on target changes)
      * filtered velocity estimate from differentiated encoder angle
      * integral separation + clamping + conditional integration (anti-windup)
      * output saturation to +/- V_supply and PWM duty quantization
    """

    def __init__(self, c: ControllerParams, d: DriveParams):
        self.c, self.d = c, d
        self.initialised = False
        self.y_c = 0.0          # unwrapped measured angle [deg]
        self.y_prev_raw = 0.0
        self.r_c = None         # latched continuous target [deg]
        self.I = 0.0            # integral term [V]
        self.w_hat = 0.0        # filtered speed estimate [rad/s]
        tau_f = 1.0 / (2 * math.pi * c.vel_filter_hz)
        self.alpha = c.Ts / (c.Ts + tau_f)
        self.last_move_deg = 0.0

    def set_target(self, target_deg: float, y_meas_deg: float):
        """Latch target in the continuous frame using the shortest path."""
        move = shortest_move_deg(target_deg, y_meas_deg, self.c.tie_band_deg)
        self.r_c = self.y_c + move
        self.last_move_deg = move
        return move

    def update(self, y_meas_deg: float):
        c = self.c
        # --- unwrap encoder (valid while |motion per sample| < 180 deg) ---
        if not self.initialised:
            self.y_c = y_meas_deg
            self.y_prev_raw = y_meas_deg
            self.initialised = True
            dy = 0.0
        else:
            dy = wrap_pm180(y_meas_deg - self.y_prev_raw)
            self.y_c += dy
            self.y_prev_raw = y_meas_deg
        if self.r_c is None:
            self.r_c = self.y_c  # hold where we are until commanded

        # --- velocity estimate: backward difference + 1st-order LPF ---
        w_raw = dy * DEG / c.Ts
        self.w_hat += self.alpha * (w_raw - self.w_hat)

        # --- PID ---
        e_deg = self.r_c - self.y_c
        e = e_deg * DEG
        P = c.Kp * e
        D = -c.Kd * self.w_hat           # derivative on measurement
        Vmax = self.d.V_supply

        I_new = self.I
        if c.Ki > 0 and abs(e_deg) < c.i_zone_deg:
            I_cand = self.I + c.Ki * e * c.Ts
            I_cand = max(-c.i_limit_V, min(c.i_limit_V, I_cand))
            u_try = P + I_cand + D
            # conditional integration: don't integrate further into saturation
            if not (abs(u_try) > Vmax and math.copysign(1, u_try) == math.copysign(1, e)):
                I_new = I_cand
        self.I = I_new

        u = P + self.I + D
        u_sat = max(-Vmax, min(Vmax, u))
        # PWM quantization (signed duty, sign -> H-bridge direction)
        levels = 2 ** self.d.pwm_bits - 1
        duty = round(u_sat / Vmax * levels) / levels
        return duty, dict(e_deg=e_deg, P=P, I=self.I, D=D, u_unsat=u)


# =============================================================================
# 6. Simulation loop
# =============================================================================
def disturbance_at(t: float, dists: List[Disturbance]) -> float:
    return sum(tau for (t_on, t_off, tau) in dists if t_on <= t < t_off)


def simulate(targets: List[Target], dists: List[Disturbance],
             motor=MotorParams(), mech=MechParams(), drive=DriveParams(),
             enc=EncoderParams(), ctrl=ControllerParams(), sim=SimConfig()):
    rng = np.random.default_rng(sim.seed)
    plant = Plant(motor, mech)
    encoder = Encoder(enc, rng)
    con = PositionController(ctrl, drive)

    n_sub = int(round(ctrl.Ts / sim.dt))
    n_ctrl = int(round(sim.t_end / ctrl.Ts))
    x = (sim.theta0_deg * DEG, sim.omega0, 0.0)

    targets = sorted(targets)
    ti = 0
    delay_buf = [0.0] * ctrl.delay_samples
    moves = []

    keys = ["t", "target", "r_c", "y_meas", "y_c", "theta", "omega", "w_hat",
            "duty", "V", "i", "tau_d", "e_meas", "e_true", "I"]
    log = {k: np.zeros(n_ctrl) for k in keys}

    for k in range(n_ctrl):
        t = k * ctrl.Ts
        y = encoder.read_deg(x[0])
        if not con.initialised:
            con.update(y)  # initialise unwrap state at t=0 (no actuation)
            con.y_c, con.y_prev_raw, con.w_hat = y, y, 0.0
        # new command?
        while ti < len(targets) and targets[ti][0] <= t + 1e-12:
            mv = con.set_target(targets[ti][1], y)
            moves.append(dict(t=targets[ti][0], target=targets[ti][1],
                              from_meas=y, move=mv))
            ti += 1
        duty, info = con.update(y)

        # actuation delay: command computed now is applied after delay_samples
        if ctrl.delay_samples > 0:
            delay_buf.append(duty)
            duty_applied = delay_buf.pop(0)
        else:
            duty_applied = duty
        V = duty_applied * drive.V_supply

        # log (state at the sampling instant)
        th_deg = x[0] * RAD
        cur_target = moves[-1]["target"] if moves else wrap_360(sim.theta0_deg)
        log["t"][k] = t
        log["target"][k] = cur_target
        log["r_c"][k] = con.r_c
        log["y_meas"][k] = y
        log["y_c"][k] = con.y_c
        log["theta"][k] = th_deg
        log["omega"][k] = x[1]
        log["w_hat"][k] = con.w_hat
        log["duty"][k] = duty_applied
        log["V"][k] = V
        log["i"][k] = x[2]
        log["tau_d"][k] = disturbance_at(t, dists)
        log["e_meas"][k] = info["e_deg"]
        log["I"][k] = info["I"]

        # integrate plant over one control period
        for s in range(n_sub):
            ts = t + s * sim.dt
            x = plant.rk4(x, V, disturbance_at(ts, dists), sim.dt)

    # True error. The controller's continuous frame is anchored to the first
    # encoder reading (= theta0 to within 1 LSB), and theta is continuous from
    # theta0 in [0,360), so the two frames coincide up to quantization.
    log["e_true"] = log["r_c"] - log["theta"]
    return log, moves


# =============================================================================
# 7. Metrics (each defined over an explicit time window)
# =============================================================================
def first_settle_time(t, e, tol):
    """Return time (relative to t[0]) after which |e| stays <= tol until the
    end of the given window; None if it never settles in the window."""
    bad = np.abs(e) > tol
    if bad[-1]:
        return None
    if not bad.any():
        return 0.0
    last_bad = np.nonzero(bad)[0][-1]
    return t[last_bad + 1] - t[0]


def compute_metrics(log, moves, dists, t_end, tol_deg=0.25, min_hold_s=0.1):
    """
    For each commanded move (segment from command time to next command):
      transition window : [t_cmd, t_cmd + t_settle]
      hold window       : [t_cmd + t_settle, first disturbance onset or next cmd]
      disturbance window: [t_on, t_off]
      recovery window   : [t_off, end of segment]
    All errors use TRUE shaft angle (available in sim only); measured-error
    versions are also reported because that is all hardware will have.
    """
    t = log["t"]
    out = []
    cmd_times = [m["t"] for m in moves] + [t_end]
    for idx, m in enumerate(moves):
        t0, t1 = m["t"], cmd_times[idx + 1]
        seg = (t >= t0) & (t < t1)
        seg_d = [d for d in dists if t0 <= d[0] < t1]
        t_hold_end = seg_d[0][0] if seg_d else t1
        win = (t >= t0) & (t < t_hold_end)
        tt, ee = t[win], log["e_true"][win]
        th = log["theta"][win]
        move = m["move"]
        s = 1.0 if move >= 0 else -1.0

        ts = first_settle_time(tt, ee, tol_deg)
        res = dict(t_cmd=t0, target=m["target"], from_meas=round(m["from_meas"], 3),
                   commanded_move_deg=round(move, 3))
        # actual displacement over the move (start -> end of hold window)
        res["actual_displacement_deg"] = round(th[-1] - th[0], 3)
        res["direction_correct"] = bool(np.sign(th[-1] - th[0]) == s
                                        or abs(move) < tol_deg)
        # 90% rise time
        prog = s * (th - th[0])
        reached = np.nonzero(prog >= 0.9 * abs(move))[0]
        res["t90_s"] = round(tt[reached[0]] - t0, 4) if (len(reached) and abs(move) > 1) else None
        res["settling_time_s"] = None if ts is None else round(ts, 4)
        overs = max(0.0, float(np.max(-s * ee)))
        res["overshoot_deg"] = round(overs, 4)
        res["overshoot_pct"] = round(100 * overs / abs(move), 3) if abs(move) > 1 else None
        res["peak_speed_deg_s"] = round(float(np.max(np.abs(log["omega"][win]))) * RAD, 1)
        res["peak_current_A"] = round(float(np.max(np.abs(log["i"][win]))), 3)
        if ts is not None:
            tr = win & (t < t0 + ts)
            res["transition_MAE_deg"] = round(float(np.mean(np.abs(log["e_true"][tr]))), 3) if tr.any() else 0.0
            hold = win & (t >= t0 + ts)
            if (t_hold_end - (t0 + ts)) >= min_hold_s:
                eh = log["e_true"][hold]
                res["hold_window_s"] = [round(t0 + ts, 3), round(t_hold_end, 3)]
                res["hold_MAE_deg"] = round(float(np.mean(np.abs(eh))), 4)
                res["hold_max_abs_err_deg"] = round(float(np.max(np.abs(eh))), 4)
                res["hold_meas_MAE_deg"] = round(float(np.mean(np.abs(log["e_meas"][hold]))), 4)
                res["hold_duty_rms"] = round(float(np.sqrt(np.mean(log["duty"][hold] ** 2))), 4)
                dd = np.diff(log["duty"][hold])
                res["hold_duty_changes_per_s"] = round(float(np.count_nonzero(dd)) / (t_hold_end - t0 - ts), 1)
                # "late hold": last 0.25 s before the window ends. The early hold
                # window starts exactly when |e| first stays below tol, so its
                # max is ~tol by construction; this window is the true steady state.
                late = hold & (t >= t_hold_end - 0.25)
                el = log["e_true"][late]
                res["late_hold_MAE_deg"] = round(float(np.mean(np.abs(el))), 4)
                res["late_hold_max_abs_err_deg"] = round(float(np.max(np.abs(el))), 4)
        # disturbance metrics
        dres = []
        for (t_on, t_off, tau) in seg_d:
            dw = (t >= t_on) & (t < t_off)
            ed = log["e_true"][dw]
            last = (t >= t_off - 0.25 * (t_off - t_on)) & (t < t_off)
            rec = (t >= t_off) & (t < t1)
            er = log["e_true"][rec]
            ts_under = first_settle_time(t[dw], ed, tol_deg)
            ts_rec = first_settle_time(t[rec], er, tol_deg) if rec.any() else None
            dres.append(dict(
                window_s=[t_on, t_off], torque_Nm=tau,
                peak_dev_during_deg=round(float(np.max(np.abs(ed))), 4),
                mean_err_last25pct_of_window_deg=round(float(np.mean(log["e_true"][last])), 4),
                back_in_tol_under_load_after_s=None if ts_under is None else round(ts_under, 4),
                peak_dev_after_release_deg=round(float(np.max(np.abs(er))), 4) if rec.any() else None,
                recovery_time_after_release_s=None if ts_rec is None else round(ts_rec, 4),
                peak_current_during_A=round(float(np.max(np.abs(log["i"][dw]))), 3),
                final_abs_err_deg=round(float(abs(er[-1])), 4) if rec.any() else None,
            ))
        if dres:
            res["disturbances"] = dres
        out.append(res)
    return out


# =============================================================================
# 8. Linear analysis helper (for gain justification)
# =============================================================================
def linear_char_roots(motor=MotorParams(), mech=MechParams(), ctrl=ControllerParams()):
    """
    Continuous-time, unsaturated, L->0, friction->viscous-only approximation:
        J th'' + B th' = Kv V + tau_d,   Kv = N Kt / R,  B = b + N^2 Kt Ke / R
        V = Kp e + Ki int(e) - Kd th'
    Closed loop characteristic polynomial:
        J s^3 + (B + Kv Kd) s^2 + Kv Kp s + Kv Ki
    """
    J = mech.J_eq(motor)
    Kv = mech.N * motor.Kt / motor.R
    B = mech.b_eq(motor) + mech.N ** 2 * motor.Kt * motor.Ke / motor.R
    if ctrl.Ki > 0:
        poly = [J, B + Kv * ctrl.Kd, Kv * ctrl.Kp, Kv * ctrl.Ki]
    else:
        poly = [J, B + Kv * ctrl.Kd, Kv * ctrl.Kp]
    return dict(J_eq=J, Kv=Kv, B_eq=B, roots=np.roots(poly),
                stall_torque_out=Kv * DriveParams().V_supply,
                no_load_speed_out_deg_s=DriveParams().V_supply * Kv / B * RAD,
                pd_static_stiffness_Nm_per_deg=Kv * ctrl.Kp * DEG)


# =============================================================================
# 9. Plotting
# =============================================================================
def _wrapped_line(t, a):
    """Insert NaN at 360->0 jumps so wrapped angle plots don't draw verticals."""
    a = np.asarray(a, float).copy()
    jumps = np.abs(np.diff(a)) > 180
    a[1:][jumps] = np.nan
    return a


def plot_run(log, dists, title, path):
    t = log["t"]
    fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    ax[0].plot(t, _wrapped_line(t, log["target"]), "k--", lw=1.2, label="commanded (0-360)")
    ax[0].plot(t, _wrapped_line(t, log["y_meas"]), "C0", lw=1.2, label="encoder output (0-360)")
    ax[0].set_ylabel("angle [deg]"); ax[0].set_ylim(-10, 370)
    ax[0].legend(loc="upper right"); ax[0].set_title(title)
    ax[1].plot(t, log["e_true"], "C3", lw=1, label="true error")
    ax[1].plot(t, log["e_meas"], "C1", lw=0.8, alpha=0.7, label="measured error")
    ax[1].set_ylabel("error [deg]"); ax[1].set_yscale("symlog", linthresh=0.1)
    ax[1].legend(loc="upper right")
    ax[2].plot(t, log["omega"] * RAD, "C2", lw=1)
    ax[2].set_ylabel("output speed [deg/s]")
    ax[3].plot(t, log["duty"], "C4", lw=0.8, label="duty (signed)")
    ax3b = ax[3].twinx()
    ax3b.plot(t, log["tau_d"], "C7", lw=1.5, label="disturbance torque")
    ax3b.set_ylabel("disturbance [N m]")
    ax[3].set_ylabel("normalized drive"); ax[3].set_xlabel("time [s]")
    ax[3].set_ylim(-1.1, 1.1)
    h1, l1 = ax[3].get_legend_handles_labels(); h2, l2 = ax3b.get_legend_handles_labels()
    ax[3].legend(h1 + h2, l1 + l2, loc="upper right")
    for a in ax:
        a.grid(alpha=0.3)
        for (t_on, t_off, _) in dists:
            a.axvspan(t_on, t_off, color="0.85", zorder=0)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_pd_vs_pid(logs, labels, window, path):
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for lg, lb in zip(logs, labels):
        m = (lg["t"] >= window[0]) & (lg["t"] <= window[1])
        ax[0].plot(lg["t"][m], lg["e_true"][m], lw=1.2, label=lb)
        ax[1].plot(lg["t"][m], lg["duty"][m], lw=0.9, label=lb)
        ax[2].plot(lg["t"][m], lg["i"][m], lw=0.9, label=lb)
    m = (logs[0]["t"] >= window[0]) & (logs[0]["t"] <= window[1])
    ax0b = ax[0].twinx(); ax0b.plot(logs[0]["t"][m], logs[0]["tau_d"][m], "0.5", lw=1)
    ax0b.set_ylabel("disturbance [N m]")
    ax[0].set_ylabel("true error [deg]"); ax[1].set_ylabel("duty")
    ax[2].set_ylabel("current [A]"); ax[2].set_xlabel("time [s]")
    ax[0].set_title("Disturbance rejection while holding: PD vs PID")
    for a in ax:
        a.grid(alpha=0.3); a.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# =============================================================================
# 10. Scenarios
# =============================================================================
MAIN_TARGETS: List[Target] = [
    (0.0, 90.0),    # T1: 0 -> 90        (+90)
    (1.0, 350.0),   # T2: 90 -> 350      (-100, shortest path is reverse)
    (2.0, 10.0),    # T3: 350 -> 10      (+20 through 0/360 wrap)
    (3.0, 190.0),   # T4: 10 -> 190      (exact 180 deg tie -> forward by rule)
    (6.0, 300.0),   # T6: 190 -> 300     (+110)
    (7.0, 45.0),    # T7: 300 -> 45      (+105 through wrap)
]
MAIN_DISTS: List[Disturbance] = [
    (4.0, 5.0, 0.5),  # +0.5 N m push for 1 s while holding 190 deg
]


def batch_random_moves(n=100, seed=1, ctrl=ControllerParams(), tol=0.25):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        th0 = float(rng.uniform(0, 360)); tgt = float(rng.uniform(0, 360))
        log, moves = simulate([(0.0, tgt)], [], ctrl=ctrl,
                              sim=SimConfig(t_end=1.0, theta0_deg=th0, seed=k))
        expected = wrap_pm180(tgt - th0)
        mv = moves[0]["move"]
        disp = log["theta"][-1] - log["theta"][0]
        ts = first_settle_time(log["t"], log["e_true"], tol)
        s = 1 if mv >= 0 else -1
        rows.append(dict(theta0=th0, target=tgt, expected_shortest=expected, move=mv,
                         displacement=disp,
                         path_ok=abs(abs(mv) - abs(expected)) < 0.6 and abs(disp - mv) < 0.5,
                         settle=ts,
                         overshoot=max(0.0, float(np.max(-s * log["e_true"]))),
                         final_err=float(abs(log["e_true"][-1])),
                         hold_mae=float(np.mean(np.abs(log["e_true"][log["t"] >= 0.8])))))
    return rows


def disturbance_capacity_sweep(torques=(0.5, 1.0, 2.0, 4.0, 5.5, 7.0), ctrl=ControllerParams()):
    """Hold 90 deg, apply each torque for 1.0 s (t=0.5..1.5), observe."""
    rows = []
    for tau in torques:
        log, _ = simulate([(0.0, 90.0)], [(0.5, 1.5, tau)], ctrl=ctrl,
                          sim=SimConfig(t_end=2.5, theta0_deg=90.0))
        dw = (log["t"] >= 0.5) & (log["t"] < 1.5)
        last = (log["t"] >= 1.25) & (log["t"] < 1.5)
        rows.append(dict(torque=tau,
                         peak_dev=float(np.max(np.abs(log["e_true"][dw]))),
                         err_end_of_push=float(np.mean(log["e_true"][last])),
                         duty_end_of_push=float(np.mean(log["duty"][last])),
                         current_end_of_push=float(np.mean(log["i"][last])),
                         final_err=float(abs(log["e_true"][-1]))))
    return rows


def main(outdir="out"):
    os.makedirs(outdir, exist_ok=True)
    results = {}
    pid = ControllerParams()
    pd = replace(pid, Ki=0.0)

    lin_pid = linear_char_roots(ctrl=pid); lin_pd = linear_char_roots(ctrl=pd)
    results["linear_model"] = dict(
        J_eq=lin_pid["J_eq"], Kv=lin_pid["Kv"], B_eq=lin_pid["B_eq"],
        stall_torque_output_Nm=lin_pid["stall_torque_out"],
        no_load_speed_output_deg_s=lin_pid["no_load_speed_out_deg_s"],
        pid_roots=[str(np.round(r, 2)) for r in lin_pid["roots"]],
        pd_roots=[str(np.round(r, 2)) for r in lin_pd["roots"]],
        pd_static_stiffness_Nm_per_deg=lin_pd["pd_static_stiffness_Nm_per_deg"],
        encoder_lsb_deg=360 / 2 ** 14)

    log_pid, mv_pid = simulate(MAIN_TARGETS, MAIN_DISTS, ctrl=pid)
    log_pd, mv_pd = simulate(MAIN_TARGETS, MAIN_DISTS, ctrl=pd)
    results["main_PID"] = compute_metrics(log_pid, mv_pid, MAIN_DISTS, 8.0)
    results["main_PD"] = compute_metrics(log_pd, mv_pd, MAIN_DISTS, 8.0)
    plot_run(log_pid, MAIN_DISTS, "Main schedule - PID controller", f"{outdir}/main_pid.png")
    plot_run(log_pd, MAIN_DISTS, "Main schedule - PD controller", f"{outdir}/main_pd.png")
    plot_pd_vs_pid([log_pd, log_pid], ["PD", "PID"], (3.8, 5.9), f"{outdir}/pd_vs_pid_disturbance.png")

    # noise sensitivity (PID, 2 LSB rms encoder noise)
    log_n, mv_n = simulate(MAIN_TARGETS, MAIN_DISTS, ctrl=pid,
                           enc=EncoderParams(noise_lsb_rms=2.0))
    results["main_PID_noise2LSB"] = compute_metrics(log_n, mv_n, MAIN_DISTS, 8.0)
    plot_run(log_n, MAIN_DISTS, "Main schedule - PID, encoder noise 2 LSB rms", f"{outdir}/main_pid_noise.png")

    rows = batch_random_moves(ctrl=pid)
    st = np.array([r["settle"] if r["settle"] is not None else np.nan for r in rows])
    mvs = np.array([abs(r["move"]) for r in rows])
    results["batch_PID"] = dict(
        n=len(rows), path_ok=int(sum(r["path_ok"] for r in rows)),
        settled_within_1s=int(np.sum(~np.isnan(st))),
        settle_s_median=float(np.nanmedian(st)), settle_s_max=float(np.nanmax(st)),
        overshoot_deg_max=float(max(r["overshoot"] for r in rows)),
        hold_MAE_deg_mean=float(np.mean([r["hold_mae"] for r in rows])),
        final_err_deg_max=float(max(r["final_err"] for r in rows)),
        failures=[r for r in rows if not r["path_ok"] or r["settle"] is None])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(mvs, st, s=14)
    ax.set_xlabel("|commanded move| [deg]"); ax.set_ylabel("settling time to +/-0.25 deg [s]")
    ax.set_title("100 random start/target pairs (PID)"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/batch_settling.png", dpi=130); plt.close(fig)

    results["capacity_sweep_PID"] = disturbance_capacity_sweep(ctrl=pid)
    results["capacity_sweep_PD"] = disturbance_capacity_sweep(ctrl=pd)

    # Gain-variant study: the integral zone/clamp trade-off
    variants = {
        "PD (Ki=0)": pd,
        "PID baseline (zone 5 deg, clamp 3 V)": pid,
        "PID zone 1 deg, clamp 3 V": replace(pid, i_zone_deg=1.0),
        "PID zone 20 deg, clamp 12 V": replace(pid, i_zone_deg=20.0, i_limit_V=12.0),
    }
    vrows = []
    for name, c in variants.items():
        lg, mv = simulate(MAIN_TARGETS, MAIN_DISTS, ctrl=c)
        m = compute_metrics(lg, mv, MAIN_DISTS, 8.0)
        cap = disturbance_capacity_sweep(torques=(2.0, 4.0), ctrl=c)
        vrows.append(dict(
            variant=name,
            mean_settle_s=float(np.mean([s["settling_time_s"] for s in m])),
            max_overshoot_deg=float(max(s["overshoot_deg"] for s in m)),
            dist_0p5Nm_err_end_deg=m[3]["disturbances"][0]["mean_err_last25pct_of_window_deg"],
            dist_0p5Nm_peak_deg=m[3]["disturbances"][0]["peak_dev_during_deg"],
            err_under_2Nm_deg=cap[0]["err_end_of_push"],
            err_under_4Nm_deg=cap[1]["err_end_of_push"]))
    results["variant_study"] = vrows

    with open(f"{outdir}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


if __name__ == "__main__":
    r = main()
    print(json.dumps(r, indent=2, default=float))
