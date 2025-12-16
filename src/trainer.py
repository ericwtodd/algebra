import os
import json

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict

from .tester import Tester

def loss_fn(outputs, targets):
    """
    Computes cross-entropy loss between model outputs and targets.
    
    Args:
        outputs (torch.Tensor): Model logits of shape [batch_size, seq_len, vocab_size]
        targets (torch.Tensor): Target token IDs of shape [batch_size, seq_len]
    
    Returns:
        torch.Tensor: Scalar loss value
    """
    loss = F.cross_entropy(
        outputs.view(-1, outputs.size(-1)),
        targets.to(torch.long).view(-1)
    )
    return loss


def accuracy_fn(outputs, targets):
    """
    Computes accuracy between predicted and target tokens.
    
    Args:
        outputs (torch.Tensor): Model logits of shape [..., vocab_size]
        targets (torch.Tensor): Target token IDs of shape [...]
    
    Returns:
        tuple: (correct, total)
            - correct (int): Number of correct predictions
            - total (int): Total number of predictions
    """
    preds = outputs.argmax(-1).view(-1)
    truth = targets.to(torch.long).view(-1)
    correct = (preds == truth).sum().item()
    total = truth.size(0)
    return correct, total


class Trainer:
    """
    Handles model training loop with evaluation, checkpointing, and logging.
    
    Manages the training process:
    - Optimization and learning rate scheduling
    - Periodic evaluation
    - Checkpointing models
    - Logging to wandb and local files
    """

    def __init__(self, config):
        """
        Initialize the Trainer.
        
        Args:
            config: Configuration object containing training parameters like
                   batch_size, learning rate, evaluation frequency, etc.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tester = Tester(config)

    def train_step(self, model, task, config, optimiser, scheduler):
        """
        Executes a single training step.
        
        Samples a batch, performs forward pass, computes loss, and updates weights.
        Supports optional left-padding and final-token-only loss computation.
        
        Args:
            model: PyTorch model to train
            task: Task object for data generation
            config: Training configuration object
            optimiser: PyTorch optimizer
            scheduler: Learning rate scheduler
        
        Returns:
            float: Loss value for this training step
        """
        context_length = model.config.block_size

        # Sample training batch
        train_batch = task.sample_batch(
            batch_size=config.batch_size,
            k_shots=config.k_shots,
            max_length=context_length,
        )

        # Move input data to the correct device
        train_batch = {k: v.to(self.device)
                for k, v in train_batch.items() if isinstance(v, torch.Tensor)}

        # Apply padding if needed
        if config.leftpad:
            pad_token_id = task.pad_token_id
            input_padding = context_length - train_batch["inputs"].size(1)
            target_padding = context_length - train_batch["targets"].size(1)
            padded_inputs = F.pad(train_batch["inputs"], (input_padding, 0), value=pad_token_id)
            padded_targets = F.pad(train_batch["targets"], (target_padding, 0), value=pad_token_id)
        else:
            padded_inputs = train_batch["inputs"]
            padded_targets = train_batch["targets"]

        # Forward pass
        outputs = model(padded_inputs)
        if not isinstance(outputs, torch.Tensor):
            outputs = outputs[0]

        if config.final_token_only:
            # Only compute loss on final token
            masked_outputs = outputs[:, -1, :].reshape(-1, outputs.size(-1))
            masked_targets = padded_targets[:, -1].reshape(-1)
        else:
            # Create mask for non-padding tokens
            mask = (padded_targets != task.pad_token_id)

            # Reshape outputs and targets, applying mask
            masked_outputs = outputs.reshape(-1, outputs.size(-1))[mask.reshape(-1)]
            masked_targets = padded_targets.reshape(-1)[mask.reshape(-1)]

        # Compute loss only on non-padded positions
        loss = loss_fn(masked_outputs, masked_targets)

        # Backward pass
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        scheduler.step()

        # Compute metrics
        return loss.item()

    def fit(self, model, task, metadata, config):
        """
        Main training loop with evaluation, checkpointing, and logging.
        
        Trains the model for config.n_steps iterations with:
        - Periodic evaluation
        - Model checkpointing
        - Optional logging
        
        Args:
            model: PyTorch model to train
            task: Task object for data generation and evaluation
            metadata (dict): Metadata to save alongside training run
            config: Training configuration object containing:
                - n_steps: Total training steps
                - lr: Learning rate
                - lr_warmup_steps: Warmup duration
                - evaluation_steps: Frequency of evaluation
                - checkpoint_steps: Frequency of checkpointing
                - use_wandb: Whether to log to wandb
                - output_dir: Directory for saving outputs
                - seed: Random seed
        
        Returns:
            list: Training losses for each step
        """
        # Set random seeds for reproducibility
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)     
        torch.cuda.manual_seed(config.seed)   

        model = model.to(self.device)

        # Setup optimizer and learning rate scheduler
        optimiser = torch.optim.AdamW(
            params=model.parameters(),
            lr=config.lr
        )
        train_scheduler = CosineAnnealingLR(
            optimizer=optimiser,
            T_max=config.n_steps,
            eta_min=0
        )
        warmup_scheduler = LinearLR(optimiser,
                start_factor=1e-4, end_factor=1, total_iters=config.lr_warmup_steps)
        scheduler = SequentialLR(optimiser,
                [warmup_scheduler, train_scheduler], [config.lr_warmup_steps])

        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name,
                config=asdict(config),
                dir="./outputs"
            )

        # Create output directory (with wandb run name if available)
        output_dir = config.output_dir
        if config.use_wandb:
            output_dir = os.path.join(config.output_dir, wandb.run.name)

        # Create results directory if it doesn't exist
        os.makedirs(f'{output_dir}/progress', exist_ok=True)

        # Write metadata file
        self.write_metadata(output_dir, metadata)

        # Training loop
        losses = []
        eval_accuracy = 0.0
        best_eval_accuracy = 0.0
        pbar = tqdm(range(config.n_steps), desc="Training", unit="it")
        for step in pbar:
            log = {}
            # Periodic evaluation (more frequent early in training)
            if (step <= 10000 and step % config.evaluation_steps == 0) or (step > 10000 and step % 1000 == 0):
                eval_accuracy, summary, other_eval_stats = self.tester.evaluate(model, task)
                
                # Task-specific logging
                pbar.write(f'Step: {step}, Acc: {round(eval_accuracy * 100, 4)}; {summary}')
                log |= {
                    "eval_accuracy": eval_accuracy 
                }
                for eval_stat_name, eval_stat_value in other_eval_stats.items():
                    log[f"{eval_stat_name}"] = eval_stat_value

                # Plot and save training loss curve
                plt.figure(figsize=(3, 3))
                plt.plot(losses)
                plt.grid(True)
                plt.title('Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.savefig(f'{output_dir}/progress/train_loss.png', bbox_inches='tight', dpi=300)
                plt.close()

                # Save best model checkpoint
                os.makedirs(f'{output_dir}/models', exist_ok=True)
                if eval_accuracy > best_eval_accuracy:
                    best_eval_accuracy = eval_accuracy
                    torch.save(model.state_dict(), f'{output_dir}/models/algebra_gpt_best.pt')

            if config.checkpoint_steps and (step % config.checkpoint_steps == 0):
                torch.save(model.state_dict(), f'{output_dir}/models/algebra_gpt_{step}.pt')

            train_loss = self.train_step(model, task, config, optimiser, scheduler)
            pbar.set_postfix({'Step': step, 'Train Loss': round(train_loss, 4)})
            losses.append(train_loss)
            log |= {
                "train_loss": train_loss,
                "learning_rate": optimiser.param_groups[0]["lr"],
                "step": step
            }
            losses.append(train_loss)

            if config.use_wandb:
                wandb.log(log)

        if config.use_wandb:
            wandb.finish()

        return losses

    def write_metadata(self, output_dir, metadata, filename="metadata.json"):
        """
        Writes metadata dictionary to JSON file.
        
        Args:
            output_dir (str): Directory to save metadata file
            metadata (dict): Metadata to serialize
            filename (str): Name of output file (default: "metadata.json")
        """
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

