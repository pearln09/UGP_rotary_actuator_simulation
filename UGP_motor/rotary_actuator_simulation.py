
"""Basic software-only rotary actuator simulation.
Requires: numpy, matplotlib
Run: python rotary_actuator_simulation.py

Illustrative model only. Parameters must be identified/tuned before hardware use.
"""

import numpy as np
import matplotlib.pyplot as plt

# Simulation settings
DT = 0.001
SIM_TIME = 15.0
ENCODER_COUNTS = 16384

# Controller and plant parameters (simulation only)
KP = 0.16
KD = 0.003
MOTOR_GAIN = 450.0
DAMPING = 18.0
MAX_SPEED = 220.0
TARGET_SPEED = 20.0  # degrees/second


def wrap_deg(angle):
    return angle % 360.0


def angle_error(target, measured):
    """Signed shortest-path error in [-180, 180)."""
    return (target - measured + 180.0) % 360.0 - 180.0


def encoder_reading(angle):
    """Idealized 14-bit single-turn encoder reading."""
    count = int(round(wrap_deg(angle) / 360.0 * ENCODER_COUNTS))
    count %= ENCODER_COUNTS
    return count * 360.0 / ENCODER_COUNTS


def target_at(t):
    if t < 0.5:
        return 350.0
    if t < 2.5:
        return 10.0
    if t < 4.5:
        return 90.0
    return 180.0


def disturbance_at(t):
    return -45.0 if 6.4 <= t <= 6.9 else 0.0


def simulate():
    n = int(SIM_TIME / DT) + 1
    t = np.arange(n) * DT

    theta = np.zeros(n)
    omega = np.zeros(n)
    target = np.zeros(n)
    measured = np.zeros(n)
    pwm = np.zeros(n)

    theta[0] = 30.0

    # Initialize the commanded trajectory at the initial requested target.
    target[0] = target_at(t[0])
    measured[0] = encoder_reading(theta[0])

    for k in range(n - 1):
        # 1. Move the commanded target toward the requested target
        desired_target = target_at(t[k])
        target_error = angle_error(desired_target, target[k])

        target_step = TARGET_SPEED * DT
        target[k + 1] = wrap_deg(
            target[k] + np.clip(
                target_error, -target_step, target_step
            )
        )

        # 2. Read the simulated encoder
        measured[k] = encoder_reading(theta[k])

        # 3. Calculate position error and controller output
        err = angle_error(target[k], measured[k])
        u = KP * err - KD * omega[k]

        # Stop the motor only when position and speed are near zero
        if abs(err) < 0.4 and abs(omega[k]) < 0.5:
            u = 0.0

        u = float(np.clip(u, -1.0, 1.0))
        pwm[k] = u

        # 4. Update motor dynamics
        alpha = (
            MOTOR_GAIN * u
            - DAMPING * omega[k]
            + disturbance_at(t[k])
        )

        omega[k + 1] = np.clip(
            omega[k] + alpha * DT,
            -MAX_SPEED,
            MAX_SPEED
        )

        theta[k + 1] = wrap_deg(
            theta[k] + omega[k + 1] * DT
        )

    # Fill in the final sample
    measured[-1] = encoder_reading(theta[-1])
    pwm[-1] = pwm[-2]

    # Calculate shortest-path absolute tracking error
    abs_error = np.abs(
        [angle_error(target[k], measured[k]) for k in range(n)]
    )

    # First commanded target transition window: 0.5 to 2.5 seconds
    start_idx = np.searchsorted(t, 0.5)
    end_idx = np.searchsorted(t, 2.5)

    window_error = abs_error[start_idx:end_idx]

    print(
        "Minimum error in first transition:",
        np.min(window_error), "degrees"
    )
    print(
        "Final error before next target change:",
        abs_error[end_idx - 1], "degrees"
    )
    print("Target angle:", target[end_idx - 1], "degrees")
    print("Measured angle:", measured[end_idx - 1], "degrees")
    print("Angular velocity:", omega[end_idx - 1], "degrees/s")
    print("Motor command:", pwm[end_idx - 1])
    print("Current KP:", KP)
    print("Current KD:", KD)
    print("Mean absolute position error:", np.mean(abs_error), "degrees")
    print("Maximum absolute position error:", np.max(abs_error), "degrees")

    steady_start_idx = np.searchsorted(t, 0.5)
    mean_error_after_startup = np.mean(abs_error[steady_start_idx:])

    print(
        "Mean absolute error after startup:",
        mean_error_after_startup, "degrees"
    )

    return t, target, measured, omega, pwm, abs_error


if __name__ == "__main__":
    t, target, measured, omega, pwm, abs_error = simulate()

    # Angle tracking
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(t, target, label="Commanded target")
    ax[0].plot(t, measured, label="Encoder measurement")
    ax[0].set_ylabel("Angle (deg)")
    ax[0].set_ylim(-5, 365)
    ax[0].set_title("Rotary actuator: closed-loop simulation")
    ax[0].legend()
    ax[0].grid(True)

    # Position error
    fig_error, ax_error = plt.subplots(figsize=(10, 3))
    ax_error.plot(t, abs_error)
    ax_error.set_xlabel("Time (s)")
    ax_error.set_ylabel("Absolute error (deg)")
    ax_error.set_title("Position tracking error")
    ax_error.grid(True)

    # Angular velocity
    ax[1].plot(t, omega)
    ax[1].set_ylabel("Velocity (deg/s)")
    ax[1].grid(True)

    # Motor command
    ax[2].plot(t, pwm)
    ax[2].set_ylabel("Motor command")
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylim(-1.1, 1.1)
    ax[2].grid(True)

    plt.tight_layout()
    fig.savefig("rotary_actuator_simulation_demo.png", dpi=160)
    fig_error.tight_layout()
    plt.show()

    print(
        "AS5048A nominal resolution:",
        360.0 / ENCODER_COUNTS, "degrees/count"
    )
    print("Simulation only; not proof of real motor performance.")
