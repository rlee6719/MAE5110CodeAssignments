import numpy as np
import matplotlib.pyplot as plt

from models import bounce_ball as model
from integrators import rk4 as integrator


params = model.generate_params()

initial_state = np.array([
    1.0,
    0.0,
])

timestep = 1e-3
sim_time = 5.0


n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep

state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state


# sim loop
for step, t in enumerate(time_traj[:-1]):

    next_state = integrator(
        model.dynamics,
        t,
        state_traj[:, step],
        timestep,
        params,
    )

    next_state = model.handle_collision(
        next_state,
        params,
    )

    state_traj[:, step + 1] = next_state


# energy
kinetic_energy, potential_energy = model.calculate_energy(
    state_traj,
    params,
)

total_energy = kinetic_energy + potential_energy


# plot
plt.figure()

plt.plot(
    time_traj,
    state_traj[0],
)

plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("Bouncing ball")

plt.tight_layout()
plt.show()


# Plot energy
plt.figure()

plt.plot(
    time_traj,
    potential_energy,
    label="Potential energy",
)

plt.plot(
    time_traj,
    kinetic_energy,
    label="Kinetic energy",
)

plt.plot(
    time_traj,
    total_energy,
    label="Total energy",
)

plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Bouncing ball energy")

plt.legend()
plt.tight_layout()
plt.show()