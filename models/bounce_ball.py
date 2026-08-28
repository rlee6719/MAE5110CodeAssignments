import numpy as np

def dynamics(t, state, params):
    gravity = params["gravity"]

    height = state[0]
    velocity = state[1]

    height_derivative = velocity
    velocity_derivative = -gravity

    return np.array([
        height_derivative,
        velocity_derivative,
    ])

def handle_collision(state, params):
    gravity = params["gravity"]

    height = state[0]
    velocity = state[1]

    if height < 0.0 and velocity < 0.0:
        state = state.copy()

        energy_per_mass = (
            gravity * height
            + 0.5 * velocity**2
        )

        # init state
        state[0] = 0.0

        state[1] = np.sqrt(
            max(2.0 * energy_per_mass, 0.0)
        )

    return state

def generate_params():
    return {
        "gravity": 9.81,
        "mass": 1.0,
    }

def calculate_energy(state, params):
    gravity = params["gravity"]
    mass = params["mass"]

    height = state[0]
    velocity = state[1]

    kinetic_energy = 0.5 * mass * velocity**2
    potential_energy = mass * gravity * height

    return kinetic_energy, potential_energy