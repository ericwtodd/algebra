import os
import json
import torch
import re
from .models.model import GPT, GPTConfig, GPTConfigNoFlashAttention
from .constants import TASK_MAP

def load_metadata(dirname: str):
    with open(os.path.join(dirname, 'metadata.json'), 'r', encoding='utf-8') as f:
        return json.load(f)

def load_gpt(dirname: str, iternum: int = None, device: str = 'cuda', disable_flash_attention=False):
    """
    """
    def latest_checkpoint(dirname: str, iternum: int = None):
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
    args = load_metadata(dirname)['args']
    task_config = args['task_config']
    if 'num_symbols' in task_config:
        print(f"Trained on {args['task_name']} up to order {task_config['max_order']} with {task_config['num_symbols']} symbols")
    else:
        print(f"Trained on {args['task_name']} up to order {task_config['max_order']} with {task_config['vocab_size']} symbols")
    task_class = TASK_MAP[args['task_name']]
    return task_class(**task_config), f"{args['task_name']} to {task_config['max_order']}"