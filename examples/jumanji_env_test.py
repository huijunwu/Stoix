import jax
import jumanji
import time

def run_jumanji_test(env_name: str = 'Snake-v1', seed: int = 0, num_steps: int = 100, render: bool = True):
    """
    Runs a basic Jumanji environment loop to test Jumanji/JAX compatibility, including rendering.
    """
    print(f"Running Jumanji test for environment: {env_name}")

    # Instantiate a Jumanji environment
    env = jumanji.make(env_name)

    # Reset your (jit-able) environment
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    state, timestep = jax.jit(env.reset)(reset_key)

    # (Optional) Render the env state
    if render:
        print("Initial render...")
        env.render(state)
        time.sleep(0.5) # Give time to see the initial state

    # Interact with the (jit-able) environment
    print("Stepping through the environment...")
    for step_idx in range(num_steps):
        if timestep.last():
            print(f"Episode finished at step {step_idx}. Resetting environment.")
            key, reset_key = jax.random.split(key)
            state, timestep = jax.jit(env.reset)(reset_key)
            if render:
                env.render(state)
                time.sleep(0.5)
            continue

        action = env.action_spec.generate_value()  # Dummy action selection
        key, step_key = jax.random.split(key) # Split key for potential stochastic actions
        state, timestep = jax.jit(env.step)(state, action)

        if render:
            env.render(state)
            time.sleep(0.1) # Small delay to visualize steps

    print("Jumanji test finished.")
    env.close()

if __name__ == '__main__':
    # You can modify these parameters to test different scenarios
    run_jumanji_test(env_name='Snake-v1', seed=42, num_steps=200, render=True)
