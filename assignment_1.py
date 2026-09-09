import numpy as np
import matplotlib.pyplot as plt

from models import rimless_wheel as model
from integrators import rk4


params = model.generate_params()

# Release the wheel slightly downhill of vertical from rest
initial_state = np.array([
    0.1,  # angle (rad)
    0.0,  # angular velocity (rad/s)
])

timestep = 1e-3
sim_time = 5.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep

state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state


def find_contact_state(t, state, timestep, params, contact_direction):
    half_spoke_angle = np.pi / params["num_spokes"]

    contact_angle = (
        params["slope"]
        + contact_direction * half_spoke_angle
    )

    lower_time = 0.0
    upper_time = timestep

    # Bisect the timestep to locate the contact
    for _ in range(10):
        middle_time = 0.5 * (lower_time + upper_time)

        middle_state = rk4(
            model.dynamics,
            t,
            state,
            middle_time,
            params,
        )

        if contact_direction == 1:
            contact_reached = middle_state[0] >= contact_angle
        else:
            contact_reached = middle_state[0] <= contact_angle

        if contact_reached:
            upper_time = middle_time
        else:
            lower_time = middle_time

    contact_time = upper_time

    contact_state = rk4(
        model.dynamics,
        t,
        state,
        contact_time,
        params,
    )

    # Set the angle exactly to the contact geometry
    contact_state[0] = contact_angle

    return contact_state, contact_time


# Simulate the rimless wheel
for step, t in enumerate(time_traj[:-1]):
    current_state = state_traj[:, step]

    next_state = rk4(
        model.dynamics,
        t,
        current_state,
        timestep,
        params,
    )

    contact_direction = model.detect_contact(
        next_state,
        params,
    )

    if contact_direction != 0:
        contact_state, contact_time = find_contact_state(
            t,
            current_state,
            timestep,
            params,
            contact_direction,
        )

        reset_state = model.reset_state(
            contact_state,
            params,
            contact_direction,
        )

        # Finish the part of the timestep remaining after impact
        remaining_time = timestep - contact_time

        next_state = rk4(
            model.dynamics,
            t + contact_time,
            reset_state,
            remaining_time,
            params,
        )

    state_traj[:, step + 1] = next_state


# Plot angle
plt.figure()

plt.plot(
    time_traj,
    state_traj[0],
)

plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title("Rimless wheel angle")
plt.tight_layout()


# Plot angular velocity
plt.figure()

plt.plot(
    time_traj,
    state_traj[1],
)

plt.xlabel("Time (s)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("Rimless wheel angular velocity")
plt.tight_layout()


plt.show()