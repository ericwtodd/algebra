import os
import json
import torch
import re
from .models.model import GPT, GPTConfig, GPTConfigNoFlashAttention
from .constants import TASK_MAP

def load_metadata(dirname: str):
    """
    Load metadata from a JSON file in the specified directory.
    
    Args:
        dirname: Path to the directory containing the metadata.json file.
        
    Returns:
        dict: Parsed metadata dictionary containing model and training information.
    """
    with open(os.path.join(dirname, 'metadata.json'), 'r', encoding='utf-8') as f:
        return json.load(f)

def load_gpt(dirname: str, iternum: int = None, device: str = 'cuda', disable_flash_attention=False):
    """
    Load a GPT model from a checkpoint directory.
    
    This function loads a trained GPT model from saved checkpoints, automatically
    selecting the latest checkpoint or a specific iteration. It handles model
    configuration and weight loading.
    
    Args:
        dirname: Path to the directory containing model checkpoints and metadata.
        iternum: Optional iteration number to load. If None, loads the latest checkpoint.
                If specified, loads the most recent checkpoint up to and including this iteration.
        device: Device to load the model onto (default: 'cuda'). Can be 'cuda', 'cpu', etc.
        disable_flash_attention: If True, uses GPTConfigNoFlashAttention instead of 
                                GPTConfig (default: False).
    
    Returns:
        GPT: Loaded GPT model with weights from the selected checkpoint.
        
    Notes:
        - Checkpoints should be stored in a 'models' subdirectory within dirname
        - Checkpoint files should match the pattern '*{number}.pt' or '*best.pt'
        - The function prints the iteration number of the loaded checkpoint
    """
    def latest_checkpoint(dirname: str, iternum: int = None):
        """
        Find the latest checkpoint file in the models directory.
        
        Args:
            dirname: Base directory containing the 'models' subdirectory.
            iternum: Optional maximum iteration number to consider.
            
        Returns:
            str: Full path to the selected checkpoint file.
            
        Notes:
            - 'best.pt' checkpoints are treated as having infinite iteration number
            - Checkpoints are sorted numerically by iteration number
        """
        modeldir = os.path.join(dirname, 'models')
        all_snapshots = sorted([
            (int(m.group(1)) if m.group(1) != 'best' else float('inf'), f)
            for f in os.listdir(modeldir)
            if (m := re.match(r'.*?(\d+|best)\.pt$', f))])
        if iternum is not None:
            all_snapshots = list((i, f) for i, f in all_snapshots if i <= iternum)
        number, name = all_snapshots[-1]
        print(f'iteration: {number}')
        return os.path.join(modeldir, name)
    
    model_params = load_metadata(dirname)['model_params']
    if disable_flash_attention:
        model = GPT(GPTConfigNoFlashAttention(**model_params))
    else:
        model = GPT(GPTConfig(**model_params))
    weights = torch.load(latest_checkpoint(dirname, iternum),
                         map_location=device, weights_only=True)
    model.load_state_dict(weights)
    return model
    
def load_task(dirname: str):
    """
    Load and instantiate a task object from training metadata.
    
    This function reads the task configuration from metadata and creates an instance
    of the corresponding task class. It also prints information about the task
    configuration.
    
    Args:
        dirname: Path to the directory containing metadata.json with task configuration.
        
    Returns:
        tuple: A tuple containing:
            - task: An instantiated task object from the appropriate task class
            - description (str): A formatted string describing the task 
              (e.g., "task_name to max_order")
    
    Notes:
        - Task classes are looked up in TASK_MAP using the task_name from metadata
        - Prints training information including task name, max_order, and either
          num_symbols or vocab_size depending on which is present in the config
    """
    args = load_metadata(dirname)['args']
    task_config = args['task_config']
    if 'num_symbols' in task_config:
        print(f"Trained on {args['task_name']} up to order {task_config['max_order']} with {task_config['num_symbols']} symbols")
    else:
        print(f"Trained on {args['task_name']} up to order {task_config['max_order']} with {task_config['vocab_size']} symbols")
    task_class = TASK_MAP[args['task_name']]
    return task_class(**task_config), f"{args['task_name']} to {task_config['max_order']}"