import argparse
import torch
import os
import json

from src.models.model import GPT, GPTConfig
from src.trainer import Trainer
from src.config import TrainingParams
from src.constants import TASK_MAP
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler")


def main(args) -> None:
    
    # Setup output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup task
    task = TASK_MAP[args.task_name](**args.task_config)

    if args.task_name == 'navigation':
        vocab_size = task.vocab_size
    else:
        vocab_size = -(-task.vocab_size // 32) * 32

    # Setup model
    model_params = {
        'n_embd': args.d_model,
        'n_layer': args.n_layers,
        'n_head': args.n_heads,
        'block_size': args.block_size,
        'vocab_size': vocab_size, # round up to multiple of 32
        'positional_encoding': args.positional_encoding,
    }
    torch.manual_seed(args.seed)
    model_config = GPTConfig(**model_params)
    model = GPT(model_config)

    # Setup training parameters
    task_config_str = '-'.join(f"{k}_{v}" for k, v in args.task_config.items()) if args.task_config else "no_config"
    wandb_run_name = f"{args.task_name}-{task_config_str}-{args.d_model}dmodel-{args.d_mlp}dmlp-{args.block_size}block-{args.n_layers}layers-{args.n_heads}heads-{args.lr}lr-{args.batch_size}batch-{args.positional_encoding}pos-{args.k_shots}shots-{args.n_steps}steps-{args.seed}seed"
    training_params = TrainingParams(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        final_token_only=args.final_token_only,
        k_shots=args.k_shots,
        lr=args.lr,
        lr_warmup_steps=args.lr_warmup_steps,
        use_wandb=args.use_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_run_name=wandb_run_name,
        evaluation_steps=args.evaluation_steps,
        checkpoint_steps=args.checkpoint_steps,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        block_size=args.block_size,
        task_name=args.task_name,
        seed=args.seed
    )

    # To save in metdata.json
    metadata = { 'args': vars(args), 'model_params': model_params }

    # Train the model
    trainer = Trainer(training_params)
    losses = trainer.fit(model, task, metadata, training_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for various tasks")
    parser.add_argument("--task_name", type=str, choices=["mixrosette", "mixcyclic", "mixdihedral", "mixmonoid"])
    parser.add_argument("--task_config", type=json.loads, default={})
    parser.add_argument("--k_shots", type=int, default=12)

    # Model parameters
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--d_mlp", type=int, default=4096)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--positional_encoding", type=str, default='rope')

    # Training parameters
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--leftpad", type=bool, default=False)
    parser.add_argument("--n_steps", type=int, default=1000000)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--evaluation_steps", type=int, default=1000)
    parser.add_argument("--checkpoint_steps", type=int, default=5000)
    parser.add_argument("--final_token_only", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=)

    # Wandb parameters
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="in-context-algebra")
    parser.add_argument("--wandb_run_name", type=str, default="")

    args = parser.parse_args()
    main(args)
