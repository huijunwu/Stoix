#!/bin/bash -ex

# This script loads a trained PPO model and performs inference.
# It uses the the 'ppo_snake_inference.py' script to load the model from a specified checkpoint.

# Usage:
#   ./serve_ppo_snake.sh [CHECKPOINT_UID] [--render] [--num_episodes NUM_EPISODES]

# Arguments:
#   CHECKPOINT_UID (optional): The unique identifier of the saved model checkpoint to load.
#                              If not provided, defaults to "my_snake_ppo_model".
#   --render (optional): If present, the environment will be rendered. (Default: True)
#   --num_episodes (optional): Number of episodes to run inference for. Defaults to 1.

# Example:
#   To serve the model with the default UID and default rendering (True):
#   ./serve_ppo_snake.sh

#   To serve a model with a custom UID and explicitly render:
#   ./serve_ppo_snake.sh my_custom_snake_model --render

#   To serve a model for 5 episodes with rendering:
#   ./serve_ppo_snake.sh --num_episodes 5

# Change to the root directory of the Git repository
cd "$(git rev-parse --show-toplevel)"

# Default values
CHECKPOINT_UID="my_snake_ppo_model"
RENDER_FLAG="--render" # Set to render by default
NUM_EPISODES_ARG=""

# Parse arguments
while (( "$#" )); do
  case "$1" in
    --render)
      # If --render is explicitly passed, keep it. This is redundant if default is --render
      # but allows for future --no-render functionality.
      RENDER_FLAG="--render"
      ;;
    --num_episodes)
      if [ -n "$2" ] && ! [[ "$2" =~ ^- ]]; then
        NUM_EPISODES_ARG="--num_episodes $2"
        shift
      else
        echo "Error: Argument for --num_episodes is missing"
        exit 1
      fi
      ;;
    -*)
      echo "Error: Unsupported flag $1"
      exit 1
      ;;
    *)
      # Assume it's the CHECKPOINT_UID if not a flag
      CHECKPOINT_UID="$1"
      ;;
  esac
  shift
done

echo "Loading PPO model for inference from checkpoint UID: ${CHECKPOINT_UID}"

python examples/inference/ppo_snake_inference.py \
    --checkpoint_uid "${CHECKPOINT_UID}" \
    ${RENDER_FLAG} \
    ${NUM_EPISODES_ARG}

echo "Inference script finished."
