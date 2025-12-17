#!/bin/bash

# Supported tasks
valid_tasks=("mixrosette" "mixcyclic" "mixdihedral")

# Check if the first argument is in the list
if [[ " ${valid_tasks[@]} " =~ " $1 " ]]; then
    echo "Training $1 task"
else
    echo "Invalid option: $1"
    exit
fi

# Four layer, 8-head-per-layer transformer
# With 1024 context length
# Train on groups up-to-order 10 from-vocabulary-of-16, never assigning 0='0'
# up to 200 examples, WITHOUT chaining.
python training.py \
    --d_model=1024 --d_mlp=4096 --block_size=1024 --n_layers=4 --n_heads=8 --leftpad=false \
    --lr=1e-5 --batch_size=128 --positional_encoding=rope \
    --n_steps=200001 --lr_warmup_steps=1000 --evaluation_steps=100 --checkpoint_steps=5000 \
    --task_name="$1" --k_shots=200 --seed=999\
    --task_config='{"num_symbols":16,"max_order":10,"mix":0.7,"holdout_zero":true}'
