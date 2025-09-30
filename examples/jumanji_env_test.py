import time

import jax
import jumanji
import jumanji.wrappers
from jumanji import specs


def run_jumanji_test(
    env_name: str = "Snake-v1", seed: int = 0, num_steps: int = 100, render: bool = True
):
    """
    Runs a basic Jumanji environment loop to test Jumanji/JAX compatibility, including rendering.
    """
    print(f"Running Jumanji test for environment: {env_name}")

    # Instantiate a Jumanji environment
    env = jumanji.make(env_name)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    gym_act_space = specs.jumanji_specs_to_gym_spaces(env.action_spec)

    # Reset your (jit-able) environment
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    state, timestep = jit_reset(reset_key)
    obs, info = timestep.observation, timestep.extras

    # Interact with the (jit-able) environment
    print("Stepping through the environment...")
    while True:
        if render:
            env.render(state)

        action = gym_act_space.sample()
        state, timestep = jit_step(state, action)
        term = ~timestep.discount.astype(bool)
        trunc = timestep.last().astype(bool)
        obs, r, info = timestep.observation, timestep.reward, timestep.extras

        if term or trunc:
            key, reset_key = jax.random.split(key)
            state, timestep = jit_reset(reset_key)

    print("Jumanji test finished.")
    env.close()


def run_jumanji_test_gym(
    env_name: str = "Snake-v1", seed: int = 0, num_steps: int = 100, render: bool = True
):
    """
    Runs a basic Jumanji environment loop to test Jumanji/JAX compatibility, including rendering.
    """
    print(f"Running Jumanji test for environment: {env_name}")

    # Instantiate a Jumanji environment
    env = jumanji.make(env_name)
    gym_env = jumanji.wrappers.JumanjiToGymWrapper(env, seed=seed)

    # Reset your (jit-able) environment
    obs, info = gym_env.reset()

    # Interact with the (jit-able) environment
    print("Stepping through the environment...")
    while True:
        if render:
            gym_env.render()

        action = gym_env.action_space.sample()  # Dummy action selection
        observation, reward, term, trunc, info = gym_env.step(action)

        if term or trunc:
            obs, info = gym_env.reset()

    print("Jumanji test finished.")
    gym_env.close()


if __name__ == "__main__":
    # You can modify these parameters to test different scenarios
    run_jumanji_test(env_name="Snake-v1", seed=42, num_steps=200, render=True)
    # run_jumanji_test_gym(env_name="Snake-v1", seed=42, num_steps=200, render=True)
