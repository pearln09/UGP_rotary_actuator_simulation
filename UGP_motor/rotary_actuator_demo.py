"""Rotary actuator closed-loop simulation demo (software only).
Requires: numpy, matplotlib
Run: python rotary_actuator_prof_demo.py
All plant parameters are illustrative and must be identified before hardware use.
"""
import numpy as np
import matplotlib.pyplot as plt

DT = 0.001
SIM_TIME = 16.0
ENCODER_COUNTS = 16384

# Illustrative controller and motor parameters (not hardware-ready values)
KP = 0.16
KD = 0.003
MOTOR_GAIN = 450.0
DAMPING = 18.0
MAX_SPEED = 220.0


def wrap_deg(angle):
    return angle % 360.0


def angle_error(target, measured):
    """Signed shortest angular error in [-180, 180)."""
    return (target - measured + 180.0) % 360.0 - 180.0


def encoder_reading(angle):
    count = int(round(wrap_deg(angle) / 360.0 * ENCODER_COUNTS)) % ENCODER_COUNTS
    return count * 360.0 / ENCODER_COUNTS


def target_at(t):
    """Piecewise position commands; includes 350 -> 10 degree wraparound."""
    if t < 1.5:
        return 350.0
    if t < 4.0:
        return 10.0
    if t < 8.0:
        return 90.0
    return 180.0


def disturbance_at(t):
    """Temporary opposing disturbance, applied after reaching the 180-degree target."""
    return -120.0 if 12.0 <= t < 12.4 else 0.0


def simulate():
    n = int(SIM_TIME / DT) + 1
    t = np.arange(n) * DT
    theta = np.zeros(n)
    omega = np.zeros(n)
    target = np.zeros(n)
    measured = np.zeros(n)
    motor_command = np.zeros(n)
    disturbance = np.zeros(n)
    theta[0] = 30.0

    for k in range(n - 1):
        target[k] = target_at(t[k])
        measured[k] = encoder_reading(theta[k])
        disturbance[k] = disturbance_at(t[k])

        error = angle_error(target[k], measured[k])
        command = KP * error - KD * omega[k]
        if abs(error) < 0.4 and abs(omega[k]) < 0.5:
            command = 0.0
        command = float(np.clip(command, -1.0, 1.0))
        motor_command[k] = command

        acceleration = MOTOR_GAIN * command - DAMPING * omega[k] + disturbance[k]
        omega[k + 1] = np.clip(omega[k] + acceleration * DT, -MAX_SPEED, MAX_SPEED)
        theta[k + 1] = wrap_deg(theta[k] + omega[k + 1] * DT)

    target[-1] = target_at(t[-1])
    measured[-1] = encoder_reading(theta[-1])
    disturbance[-1] = disturbance_at(t[-1])
    motor_command[-1] = motor_command[-2]
    error_abs = np.abs([angle_error(target[i], measured[i]) for i in range(n)])

    print("ROTARY ACTUATOR — SOFTWARE SIMULATION")
    print(f"Encoder resolution: {360 / ENCODER_COUNTS:.5f} deg/count (nominal quantization)")
    print(f"Overall mean absolute tracking error: {np.mean(error_abs):.2f} deg")
    for label, lo, hi in [("350 -> 10 degree transition", 1.5, 4.0),
                          ("90-degree target interval", 4.0, 8.0),
                          ("After disturbance ends", 12.4, 16.0)]:
        mask = (t >= lo) & (t < hi)
        print(f"{label}: mean error = {np.mean(error_abs[mask]):.2f} deg")
    print("Note: simulation only; not proof of real motor performance.")
    return t, target, measured, omega, motor_command, error_abs, disturbance


if __name__ == "__main__":
    t, target, measured, omega, command, error_abs, disturbance = simulate()
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(t, target, label="Commanded position")
    axes[0].plot(t, measured, label="Encoder measurement")
    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_ylim(-5, 365)
    axes[0].set_title("Rotary actuator closed-loop simulation")
    axes[0].legend(loc="best")
    axes[0].grid(True)

    axes[1].plot(t, error_abs)
    axes[1].set_ylabel("Abs. error (deg)")
    axes[1].grid(True)

    axes[2].plot(t, omega)
    axes[2].set_ylabel("Velocity (deg/s)")
    axes[2].grid(True)

    axes[3].plot(t, command, label="Motor command")
    axes[3].set_ylabel("Command")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylim(-1.1, 1.1)
    axes[3].grid(True)

    for ax in axes:
        ax.axvspan(12.0, 12.4, alpha=0.15, color="red")
    axes[3].legend(loc="best")
    fig.tight_layout()
    fig.savefig("rotary_actuator_prof_demo.png", dpi=180)
    plt.show()
