import dataclasses
import os
import json
import torch
from dataclasses import dataclass, asdict, replace
from datetime import datetime
from typing import Callable
import torch.nn as nn


@dataclass
class TrainingParams:

    n_steps: int
    batch_size: int
    lr_warmup_steps: int = 500
    seed: int = 42
    lr: float = 1e-5
    loss_fn: Callable = None
    accuracy_fn: Callable = None
    final_token_only: bool = False
    use_wandb: bool = True
    wandb_entity: str = ""
    wandb_project: str = ""
    wandb_run_name: str = ""
    evaluation_steps: int = 200
    evaluation_size: int = 500
    evaluation_length: int = 500
    checkpoint_steps: int = 0
    k_shots: int = 200
    leftpad: bool = False
    output_dir: str = "outputs"

    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 8
    block_size: int = 128
    task_name: str = "facts"

@dataclass
class BaseConfig:

    # Task Parameter
    n_samples: int = -1
    train_size: float = 0.8

    # Model Parameters
    context_length: int = -1
    n_layers: int = -1
    d_model: int = -1
    n_heads: int = -1
    d_mlp: int = -1
    d_vocab: int = -1
    attention_dir: str = "causal"
    act_fn: str = "gelu"
    positional_embedding_type: str = "fire"
    attn_only: bool = False

    # Training Parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_steps: int = -1
    batch_size: int = -1
    lr: float = 4e-4
    lr_warmup_steps: int = 1024
    lr_decay_iters: int = 1024
    beta_1: float = 0.9
    beta_2: float = 0.99
    wd: float = 0.01
    evaluation_interval: int = 64

    # I/O
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    checkpoint_interval: int = 32

    # Weights and Biases
    use_wandb: bool = True
    wandb_entity: str = ""
    wandb_project: str = ""
    wandb_name: str = ""
    wandb_group: str = ""

    train_length: int = 4
    test_length: int = 8

    use_binaries: bool = False
    max_number: int = 99
    repeat_numbers: bool = False


def get_configs(
    config: BaseConfig,
    attr_name: str,
    values: list,
    wandb_prefix: str = "",
):
    configs = []
    for value in values:

        # add sweep parameter
        config_dict = asdict(config)
        config_dict[attr_name] = value
        config = replace(config, **config_dict)

        # replace wandb_name
        if not isinstance(value, str):
            value = round(value, 6)

        # define wandb_name
        wandb_name = ""
        if wandb_prefix:
            wandb_name = f"{wandb_prefix}_"
        wandb_name += f"{attr_name}_{value}_"
        wandb_name += datetime.now().strftime("%Y%m%d%H%M%S%f")
        config = replace(config, wandb_name=wandb_name)
        configs.append(config)
    return configs


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def save_config(config: dataclass, checkpoint_dir: str):
    filepath = os.path.join(checkpoint_dir, "config.json")
    with open(filepath, 'w') as f:
        json.dump(config, f, cls=EnhancedJSONEncoder, indent=4)
