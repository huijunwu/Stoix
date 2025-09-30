import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
import hydra # Required for hydra.utils.instantiate
import os
from typing import Any # Import Any

# Assuming these imports are available in your environment
from stoix.utils.checkpointing import Checkpointer
from stoix.networks.base import FeedForwardActor as Actor
from stoix.base_types import ActorCriticParams # This type holds actor and critic parameters
from stoix.utils import make_env as environments # To create the environment for observation space

def _load_model_components(
    model_name: str,
    checkpoint_uid: str,
    checkpoints_dir: str,
) -> tuple[Any, Any, Any]:
    """Loads the checkpoint, restores parameters, and reconstructs the actor network."""
    print(f"Loading checkpoint for model '{model_name}' with UID '{checkpoint_uid}'...")
    checkpointer = Checkpointer(
        model_name=model_name,
        rel_dir=checkpoints_dir,
        checkpoint_uid=checkpoint_uid,
    )
    restored_params, _ = checkpointer.restore_params(TParams=ActorCriticParams)
    actor_params = restored_params.actor_params
    print("Actor parameters loaded successfully.")

    cfg = checkpointer.get_cfg()
    print("Original training configuration loaded from checkpoint metadata.")

    actor_torso = hydra.utils.instantiate(cfg.custom_metadata.network.actor_network.pre_torso)
    num_actions = cfg.custom_metadata.system.action_dim
    actor_action_head = hydra.utils.instantiate(
        cfg.custom_metadata.network.actor_network.action_head, action_dim=num_actions
    )
    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)
    print("Actor network reconstructed.")
    return actor_params, actor_network, cfg

def _create_environment(
    env_name: str,
    scenario_name: str,
    num_rows: int,
    num_cols: int,
) -> Any:
    """Creates and configures the Jumanji environment."""
    env_config = OmegaConf.create({
        "env": {
            "env_name": env_name,
            "observation_attribute": "grid",
            "multi_agent": False,
            "scenario": {"name": scenario_name},
            "kwargs": {"num_rows": num_rows, "num_cols": num_cols},
            "wrapper": {"_target_": "stoa.FlattenObservationWrapper"}
        }
    })
    env, _ = environments.make(config=env_config)
    return env

def _run_episode(
    env: Any,
    actor_network: Any,
    actor_params: Any,
    key: jax.random.PRNGKey,
    render: bool,
    episode_idx: int,
) -> tuple[jax.random.PRNGKey, float]:
    """Runs a single episode of inference and returns the final episode return."""
    print(f"--- Starting Episode {episode_idx + 1} ---")
    key, reset_key = jax.random.split(key)
    batched_reset_key = jax.tree_util.tree_map(lambda x: x[None, ...], reset_key)
    env_state, timestep = env.reset(batched_reset_key)

    if render:
        env.unwrapped.render(env_state.unwrapped_state)

    while not timestep.last():
        observation = jax.tree_util.tree_map(lambda x: x[None, ...], timestep.observation)
        actor_policy = actor_network.apply(actor_params, observation)
        key, action_key = jax.random.split(key)
        action = actor_policy.sample(seed=action_key)
        action = jax.tree_util.tree_map(lambda x: x.squeeze(0), action)
        env_state, timestep = env.step(env_state, action)

        if render:
            env.unwrapped.render(env_state.unwrapped_state)

    print(f"Episode {episode_idx + 1} finished. Episode return: {timestep.reward}")
    return key, float(timestep.reward)

def run_inference(
    model_name: str = "ff_ppo",
    checkpoint_uid: str = "my_snake_ppo_model",
    checkpoints_dir: str = "checkpoints",
    env_name: str = "jumanji",
    scenario_name: str = "Snake-v1",
    num_rows: int = 6,
    num_cols: int = 6,
    seed: int = 0,
    num_episodes: int = 1,
    render: bool = False,
):
    """
    Loads a trained PPO model and performs inference on a dummy observation.

    Args:
        model_name (str): The name of the model (e.g., "ff_ppo").
        checkpoint_uid (str): The unique ID of the checkpoint (e.g., "my_snake_ppo_model" or a timestamp).
        checkpoints_dir (str): The relative directory where checkpoints are stored.
        env_name (str): The name of the environment suite (e.g., "jumanji").
        scenario_name (str): The name of the environment scenario (e.g., "Snake-v1").
        num_rows (int): Number of rows in the Snake environment grid.
        num_cols (int): Number of columns in the Snake environment grid.
        seed (int): Random seed for JAX.
    """

    actor_params, actor_network, cfg = _load_model_components(
        model_name, checkpoint_uid, checkpoints_dir
    )

    env = _create_environment(env_name, scenario_name, num_rows, num_cols)

    key = jax.random.PRNGKey(seed)
    for episode_idx in range(num_episodes):
        key, episode_key = jax.random.split(key)
        key, episode_return = _run_episode(
            env, actor_network, actor_params, episode_key, render, episode_idx
        )

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PPO model inference.")
    parser.add_argument("--model_name", type=str, default="ff_ppo",
                        help="The name of the model (e.g., 'ff_ppo').")
    parser.add_argument("--checkpoint_uid", type=str, default="my_snake_ppo_model",
                        help="The unique ID of the checkpoint to load.")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints",
                        help="The relative directory where checkpoints are stored.")
    parser.add_argument("--env_name", type=str, default="jumanji",
                        help="The name of the environment suite (e.g., 'jumanji').")
    parser.add_argument("--scenario_name", type=str, default="Snake-v1",
                        help="The name of the environment scenario (e.g., 'Snake-v1').")
    parser.add_argument("--num_rows", type=int, default=6,
                        help="Number of rows in the Snake environment grid.")
    parser.add_argument("--num_cols", type=int, default=6,
                        help="Number of columns in the Snake environment grid.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for JAX.")
    parser.add_argument("--num_episodes", type=int, default=1,
                        help="Number of episodes to run inference for.")
    parser.add_argument("--render", action="store_true",
                        help="Whether to render the environment.")

    args = parser.parse_args()

    run_inference(
        model_name=args.model_name,
        checkpoint_uid=args.checkpoint_uid,
        checkpoints_dir=args.checkpoints_dir,
        env_name=args.env_name,
        scenario_name=args.scenario_name,
        num_rows=args.num_rows,
        num_cols=args.num_cols,
        seed=args.seed,
        num_episodes=args.num_episodes,
        render=args.render,
    )
