#!/bin/bash -ex

# This script trains a PPO model for the Jumanji Snake environment.
# It enables model checkpointing and allows specifying a unique ID for the checkpoint.

# Usage:
#   ./train_ppo_snake.sh [CHECKPOINT_UID]

# Arguments:
#   CHECKPOINT_UID (optional): A unique identifier for the saved model checkpoint.
#                              If not provided, defaults to "my_snake_ppo_model".

# Example:
#   To train and save a model with the default UID:
#   ./train_ppo_snake.sh

#   To train and save a model with a custom UID:
#   ./train_ppo_snake.sh my_custom_snake_model

# Change to the root directory of the Git repository
cd "$(git rev-parse --show-toplevel)"

# Default checkpoint UID if not provided
CHECKPOINT_UID=${1:-"my_snake_ppo_model"}

echo "Starting PPO training for Jumanji Snake environment..."
echo "Model will be saved with checkpoint UID: ${CHECKPOINT_UID}"

python stoix/systems/ppo/anakin/ff_ppo.py \
    env=jumanji/snake \
    logger.checkpointing.save_model=true \
    logger.checkpointing.save_args.checkpoint_uid="${CHECKPOINT_UID}"

echo "Training complete. Model saved to checkpoints/ff_ppo/${CHECKPOINT_UID}/"
