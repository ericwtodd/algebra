from dataclasses import dataclass

@dataclass
class TrainingParams:

    # Task Parameters
    task_name: str
    task_config: dict
    k_shots: int = 200

    # Model Parameters
    n_layers: int = 4
    n_heads: int = 8
    d_model: int = 1024
    d_mlp: int = 4096
    block_size: int = 1024  # Context Window
    positional_encoding: str = 'rope'

    # Training Parameters
    output_dir: str = "outputs"
    leftpad: bool = False
    batch_size: int = 128
    n_steps: int = 200001
    lr: float = 1e-5
    lr_warmup_steps: int = 1000
    evaluation_steps: int = 100
    checkpoint_steps: int = 5000
    final_token_only: bool = False
    seed: int = 9999

    evaluation_size: int = 500
    
    # Wandb parameters
    use_wandb: bool = True
    wandb_entity: str = ""
    wandb_project: str = ""
    wandb_run_name: str = ""
