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
    restitution = params["restitution"]

    height = state[0]
    velocity = state[1]

    if height < 0.0 and velocity < 0.0:
        state = state.copy()

        # Mechanical energy per unit mass immediately
        # before correcting the ground penetration
        energy_per_mass = (
            gravity * height
            + 0.5 * velocity**2
        )

        # Move ball back to ground
        state[0] = 0.0

        # Calculate speed the ball should have at impact
        impact_speed = np.sqrt(
            max(2.0 * energy_per_mass, 0.0)
        )

        # Apply coefficient of restitution
        state[1] = restitution * impact_speed

    return state

def generate_params():
    return {
        "gravity": 9.81,
        "mass": 1.0,
        "restitution": 0.8,
    }

def calculate_energy(state, params):
    gravity = params["gravity"]
    mass = params["mass"]

    height = state[0]
    velocity = state[1]

    kinetic_energy = 0.5 * mass * velocity**2
    potential_energy = mass * gravity * height

    return kinetic_energy, potential_energy