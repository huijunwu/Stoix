#!/bin/bash -ex

# This script runs a basic Jumanji environment test with rendering.
# It helps verify Jumanji/JAX compatibility.

# Usage:
#   ./run_jumanji_test.sh

# Change to the root directory of the Git repository
cd "$(git rev-parse --show-toplevel)"

echo "Running Jumanji environment test..."

# Set Matplotlib backend for interactive rendering.
# 'TkAgg' is a common interactive backend. Other options include 'Qt5Agg', 'GTK3Agg', etc.
# Ensure the necessary backend libraries are installed on your system.
export MPLBACKEND=TkAgg

python examples/jumanji_env_test.py

echo "Jumanji environment test finished."