import torch
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.intervention_utils import cache_activations
import matplotlib.pyplot as plt


# =============================================================================
# Data Generation for Probe Training
# =============================================================================

def sample_balanced_batch_for_symbol(task, batch_size, target_symbol, num_shots=20, probe_type='closure'):
    """
    Generate balanced batch for training a probe to detect if target_symbol
    is in the closure of the query (symbols appearing in facts that share
    elements with the query).
    
    Args:
        task: Task object with sample_run(), etc.
        batch_size: Number of positive examples (creates batch_size*2 total)
        target_symbol: Symbol to detect in closure
        num_shots: Number of facts per prompt
    
    Returns:
        Dictionary with:
            'prompts': [batch_size*2, seq_len-1]
            'probe_labels': [batch_size*2] where 1 = symbol in closure, 0 = not
            'closures': list of closure sets for each example
            'group_vocabs': list of vocab for the query's group
            'full_vocab': list of full vocab strings for each example
    """
    def get_closure(sequence):
        """
        Get the closure: symbols appearing in facts that share query elements.
        
        1. Finding facts that share symbols with the query
        2. Computing the closure over all symbols in those facts
        """
        facts = sequence.split(',')
        query = facts[-1]
        
        share_symbol = [fact for fact in facts[1:-1] if query[0] in fact or query[1] in fact]
        
        # Get all unique symbols from these facts (including answers)
        closure_set = list(set(''.join(share_symbol).replace('=', '')))
        return closure_set
    
    def get_elimination(sequence):
        """
        Get the elimination set: answer symbols from left and right cosets.
        
        These are the symbols that appear as answers in the cosets,
        representing what gets "eliminated" or "cancelled out".
        """
        facts = sequence.split(',')
        query = facts[-1]
        
        share_a_on_left = [fact for fact in facts[1:-1] if fact[0] == query[0]]
        share_b_on_right = [fact for fact in facts[1:-1] if fact[1] == query[1]]
        
        share_symbol_slots = share_a_on_left + share_b_on_right
        
        # Only the answer symbols (after '=')
        elimination_set = list(set([fact.split('=')[1] for fact in share_symbol_slots if '=' in fact]))
        return elimination_set
    
    def get_query_group_vocab(sequence, vocab_string, orders):
        """Extract the vocab for the group that the query belongs to."""
        facts = sequence.split(',')
        query = facts[-1]
        query_symbol = query[0]  # Use first symbol of query to identify group
        
        # Split vocab by groups
        start = 0
        for order in orders:
            vocab_part = vocab_string[start:start+order]
            if query_symbol in vocab_part:
                return list(vocab_part)
            start += order
        
        return []  # Shouldn't happen if data is valid
    
    all_prompts = []
    all_labels = []
    all_closures = []
    all_group_vocabs = []
    all_full_vocabs = []

    if probe_type == 'closure':
        pos_label_fn = get_closure
    elif probe_type == 'elimination':
        pos_label_fn = get_elimination
    else:
        raise ValueError("probe type unspecified")
    
    for _ in range(batch_size):
        # Positive: resample until target_symbol is in the closure/elimination set
        while True:
            prompt_pos, groups_pos, orders_pos, vocab_pos = task.sample_run(
                k_shots=num_shots, hold_out=0
            )
            closure_pos = pos_label_fn(prompt_pos)
            if target_symbol in closure_pos:
                break
        
        group_vocab_pos = get_query_group_vocab(prompt_pos, vocab_pos, orders_pos)
        
        # Negative: resample until target_symbol is NOT in the closure/elimination set
        while True:
            prompt_neg, groups_neg, orders_neg, vocab_neg = task.sample_run(
                k_shots=num_shots, hold_out=0
            )
            closure_neg = pos_label_fn(prompt_neg)
            if target_symbol not in closure_neg:
                break
        
        group_vocab_neg = get_query_group_vocab(prompt_neg, vocab_neg, orders_neg)
        
        all_prompts.extend([prompt_pos, prompt_neg])
        all_labels.extend([1, 0])
        all_closures.extend([closure_pos, closure_neg])
        all_group_vocabs.extend([group_vocab_pos, group_vocab_neg])
        all_full_vocabs.extend([vocab_pos, vocab_neg])
    
    # Convert and shuffle
    prompts_tensor = task.tensor_from_expression(all_prompts)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    rand_inds = torch.randperm(len(all_labels))
    
    shuffled_closures = [all_closures[i] for i in rand_inds]
    shuffled_group_vocabs = [all_group_vocabs[i] for i in rand_inds]
    shuffled_full_vocabs = [all_full_vocabs[i] for i in rand_inds]
    
    return {
        'prompts': prompts_tensor[rand_inds, :-1],
        'probe_labels': labels_tensor[rand_inds],
        'closures': shuffled_closures,           # Closure sets
        'group_vocabs': shuffled_group_vocabs,   # Query group's vocab
        'full_vocab': shuffled_full_vocabs       # Full vocab string
    }


# =============================================================================
# Binary Probe
# =============================================================================

class BinaryProbe(torch.nn.Module):
    """Simple linear binary probe."""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1, bias=False)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train_binary_probe(X, y, num_epochs=1000, lr=0.001, test_size=0.2, 
                       verbose=True, eval_interval=10):
    """
    Train a binary probe on subspace-projected activations.
    
    Args:
        X: Data tensor [n_samples, subspace_dim]
        y: Binary labels [n_samples]
        num_epochs: Number of training epochs
        lr: Learning rate
        test_size: Fraction for test set
        verbose: Show progress bar
        eval_interval: Evaluate test set every N epochs
    
    Returns:
        Dictionary with trained probe, metrics, and probe direction
    """
    # Convert to tensors
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    if not isinstance(y, torch.Tensor):
        y = torch.FloatTensor(y)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Setup
    probe = BinaryProbe(X.shape[1])
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.001)
    
    losses = []
    train_accs = []
    test_accs = []
    best_test_acc = 0
    best_probe_state = None
    
    pbar = tqdm(range(num_epochs), desc="Probe Training", disable=not verbose)
    
    probe.train()
    for epoch in pbar:
        optimizer.zero_grad()
        outputs = probe(X_train).squeeze()
        loss = loss_fn(outputs, y_train.float())
        losses.append(loss.item())
        
        # Training accuracy
        with torch.no_grad():
            train_preds = (outputs > 0.5).float()
            train_acc = (train_preds == y_train).float().mean().item()
            train_accs.append(train_acc)
        
        loss.backward()
        optimizer.step()
        
        # Evaluate on test set
        if epoch % eval_interval == 0:
            probe.eval()
            with torch.no_grad():
                test_outputs = probe(X_test).squeeze()
                test_preds = (test_outputs > 0.5).float()
                test_acc = (test_preds == y_test).float().mean().item()
                test_accs.append(test_acc)
                
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_probe_state = probe.state_dict().copy()
            probe.train()
        
        if verbose:
            current_test_acc = test_accs[-1] if test_accs else 0
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Train": f"{train_acc:.4f}",
                "Test": f"{current_test_acc:.4f}"
            })
    
    # Final evaluation
    probe.eval()
    with torch.no_grad():
        train_outputs = probe(X_train).squeeze()
        test_outputs = probe(X_test).squeeze()
        
        train_preds = (train_outputs > 0.5).float()
        test_preds = (test_outputs > 0.5).float()
        
        final_train_acc = (train_preds == y_train).float().mean().item()
        final_test_acc = (test_preds == y_test).float().mean().item()
    
    probe_direction = probe.linear.weight.data.squeeze()
    
    # Load best probe state
    if best_probe_state is not None:
        probe.load_state_dict(best_probe_state)
        best_probe_direction = probe.linear.weight.data.squeeze()
    else:
        best_probe_direction = probe_direction
    
    return {
        "probe": probe,
        "losses": losses,
        "train_accs": train_accs,
        "test_accs": test_accs,
        "final_test_acc": final_test_acc,
        "best_test_acc": best_test_acc,
        "probe_direction": probe_direction,
        "best_probe_direction": best_probe_direction
    }


# =============================================================================
# Training Pipeline
# =============================================================================

# In the train_probes_for_vocabulary function, update this section:

def train_probes_for_vocabulary(model, task, subspace_proj, vocab_symbols,
                                 layer=3, component='attn', batch_size=2000, 
                                 num_shots=20, probe_epochs=2000, probe_type='closure'):
    """
    Train binary probes for each symbol in the vocabulary.
    
    Args:
        model: The language model
        task: Task object
        subspace_proj: Projection matrix into subspace [hidden_dim, subspace_dim]
        vocab_symbols: List of vocabulary symbols to train probes for
        layer: Layer to extract activations from
        component: Component type ('attn', 'mlp', 'block')
        batch_size: Batch size for data generation
        num_shots: Shots per prompt
        probe_epochs: Training epochs per probe
    
    Returns:
        Dictionary mapping symbols to probe results
    """
    
    # Setup model component
    if component == 'attn':
        model_component = model.transformer.h[layer].attn
        model_component_str = f'model.transformer.h[{layer}].attn'
    elif component == 'mlp':
        model_component = model.transformer.h[layer].mlp
        model_component_str = f'model.transformer.h[{layer}].mlp'
    elif component == 'block':
        model_component = model.transformer.h[layer]
        model_component_str = f'model.transformer.h[{layer}]'
    else:
        raise ValueError(f"Unknown component: {component}")
    
    probe_results = {}
    
    for probe_symbol in tqdm(vocab_symbols, desc="Training probes"):
        with torch.no_grad():
            batch = sample_balanced_batch_for_symbol(
                task, batch_size, probe_symbol, num_shots, probe_type=probe_type
            )
            # Remove token_idx parameter
            batch_act_cache = cache_activations(
                model, batch['prompts'], model_component, model_component_str
            )
        
        torch.cuda.empty_cache()
        
        # Project into subspace - now need to select the last token manually
        data_X = (batch_act_cache[model_component_str][:, -1, :] @ subspace_proj)
        data_Y = batch['probe_labels'].long()
        
        results = train_binary_probe(data_X, data_Y, num_epochs=probe_epochs, 
                                     lr=1e-3, verbose=False)
        
        # Store metadata
        results['batch_metadata'] = {
            'closures': batch['closures'],
            'group_vocabs': batch['group_vocabs'],
            'full_vocab': batch['full_vocab']
        }
        
        probe_results[probe_symbol] = results
    
    return probe_results

# =============================================================================
# Visualization & Validation
# =============================================================================

def remap_vocab_to_letters(vocab_symbols):
    """Remap vocab symbols (0-9a-f) to letters (a-p) for display."""
    letter_map = {
        '0': 'a', '1': 'b', '2': 'c', '3': 'd', '4': 'e', '5': 'f',
        '6': 'g', '7': 'h', '8': 'i', '9': 'j', 'a': 'k', 'b': 'l',
        'c': 'm', 'd': 'n', 'e': 'o', 'f': 'p'
    }
    return [letter_map.get(s, s) for s in vocab_symbols]

import matplotlib.pyplot as plt

# Set ICLR-style parameters at the top of your script
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['text.usetex'] = False


def plot_probe_generalization_comparison(closure_gen_matrix, elimination_gen_matrix, vocab_symbols,
                                         save_path=None):
    """Plot closure and elimination probe generalization side-by-side."""
    display_labels = remap_vocab_to_letters(vocab_symbols)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for idx, (gen_matrix, title) in enumerate([
        (closure_gen_matrix, "Closure Probes"),
        (elimination_gen_matrix, "Elimination Probes")
    ]):
        ax = axes[idx]
        im = ax.imshow(gen_matrix, vmax=1, vmin=0, cmap='PiYG')
        
        # Add text annotations
        for i in range(gen_matrix.shape[0]):
            for j in range(gen_matrix.shape[1]):
                text_color = "white" if gen_matrix[i, j] < 0.4 else "black"
                ax.text(j, i, f"{gen_matrix[i, j].item():.2f}",
                       ha="center", va="center",
                       color=text_color, fontsize=13, weight='medium')
        
        ax.set_xticks(range(len(vocab_symbols)))
        ax.set_yticks(range(len(vocab_symbols)))
        ax.set_xticklabels(display_labels, fontsize=12)
        ax.set_yticklabels(display_labels, fontsize=12)
        ax.set_xlabel("Probe Trained For Symbol", fontweight='medium', fontsize=13)
        ax.set_ylabel("Data Labeled As Symbol", fontweight='medium', fontsize=13)
        ax.set_title(title, fontweight='medium', fontsize=13)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Accuracy', fontweight='medium', fontsize=13)
        cbar.ax.tick_params(labelsize=12)
        
        ax.grid(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    
    return fig


def plot_probe_unembed_alignment_comparison(closure_probe_results, elimination_probe_results,
                                            model, task, vocab_symbols, 
                                            closure_proj, elimination_proj,
                                            save_path=None):
    """
    Plot closure and elimination probe-unembed alignment side-by-side.
    Includes special tokens (= and ,) as additional rows.
    """
    display_labels = remap_vocab_to_letters(vocab_symbols)
    
    # Add special tokens to vocab
    all_vocab_symbols = vocab_symbols + ['=', ',']
    all_display_labels = display_labels + ['=', ',']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for idx, (probe_results, subspace_proj, title) in enumerate([
        (closure_probe_results, closure_proj, "Closure Probes"),
        (elimination_probe_results, elimination_proj, "Elimination Probes")
    ]):
        # Get probe directions in subspace
        probe_dirs = torch.vstack([
            probe_results[v]['probe'].linear.weight.data 
            for v in vocab_symbols
        ])
        
        # Project back to full space
        probe_dirs_full = (probe_dirs @ subspace_proj.T)  # [num_probes, hidden_dim]
        
        # Get unembedding vectors for all symbols (including special tokens)
        vocab_indices = [task.vocab.index(s) for s in all_vocab_symbols]
        unembed_dirs = model.lm_head.weight.data[vocab_indices]  # [num_symbols+2, hidden_dim]
        
        # Compute cosine similarities
        cosine_sims = torch.zeros(len(all_vocab_symbols), len(vocab_symbols))
        for i in range(len(all_vocab_symbols)):
            for j in range(len(vocab_symbols)):
                cosine_sims[i, j] = torch.cosine_similarity(
                    unembed_dirs[i:i+1].cpu(), 
                    probe_dirs_full[j:j+1].cpu(), 
                    dim=1
                ).item()
        
        ax = axes[idx]
        im = ax.imshow(cosine_sims, vmax=1, vmin=-1, cmap='RdBu')
        
        # Add text annotations
        for i in range(cosine_sims.shape[0]):
            for j in range(cosine_sims.shape[1]):
                text_color = "white" if abs(cosine_sims[i, j]) > 0.5 else "black"
                ax.text(j, i, f"{cosine_sims[i, j]:.2f}",
                       ha="center", va="center",
                       color=text_color, fontsize=13, weight='medium')
        
        ax.set_xticks(range(len(vocab_symbols)))
        ax.set_yticks(range(len(all_vocab_symbols)))
        ax.set_xticklabels(display_labels, fontsize=12)
        ax.set_yticklabels(all_display_labels, fontsize=12)
        ax.set_xlabel("Probe Trained for Symbol", fontweight='medium', fontsize=13)
        ax.set_ylabel("Unembed Direction for Symbol", fontweight='medium', fontsize=13)
        ax.set_title(title, fontweight='medium', fontsize=13)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Cosine Similarity', fontweight='medium', fontsize=13)
        cbar.ax.tick_params(labelsize=12)
        
        ax.grid(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    
    return fig


def plot_probe_similarities_comparison(closure_probe_results, elimination_probe_results,
                                       vocab_symbols, save_path=None):
    """Plot closure and elimination probe direction similarities side-by-side."""
    display_labels = remap_vocab_to_letters(vocab_symbols)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for idx, (probe_results, title) in enumerate([
        (closure_probe_results, "Closure Probes"),
        (elimination_probe_results, "Elimination Probes")
    ]):
        probe_dirs = torch.vstack([
            probe_results[v]['probe'].linear.weight.data 
            for v in vocab_symbols
        ])
        
        cossims = torch.vstack([
            torch.cosine_similarity(probe_dirs, probe_dirs[i], dim=1) 
            for i in range(len(vocab_symbols))
        ])
        
        ax = axes[idx]
        im = ax.imshow(cossims, vmax=1, vmin=-1, cmap='RdBu')
        
        # Add text annotations
        for i in range(cossims.shape[0]):
            for j in range(cossims.shape[1]):
                text_color = "white" if cossims[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cossims[i, j].item():.2f}",
                       ha="center", va="center",
                       color=text_color, fontsize=13, weight='medium')
        
        ax.set_xticks(range(len(vocab_symbols)))
        ax.set_yticks(range(len(vocab_symbols)))
        ax.set_xticklabels(display_labels, fontsize=12)
        ax.set_yticklabels(display_labels, fontsize=12)
        ax.set_title(title, fontweight='medium', fontsize=13)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Cosine Similarity', fontweight='medium', fontsize=13)
        cbar.ax.tick_params(labelsize=12)
        
        ax.grid(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    
    return fig

def evaluate_probe_generalization(model, task, subspace_proj, probe_results, 
                                   vocab_symbols, layer=3, component='attn',
                                   batch_size=2000, num_shots=30, probe_type='closure'):
    """
    Evaluate how well each probe generalizes to detecting other symbols.
    
    Creates a confusion matrix showing probe accuracy when applied to
    data labeled for different symbols.
    
    Returns:
        Tensor [len(vocab_symbols), len(vocab_symbols)] of accuracies
    """    
    
    # Setup model component
    if component == 'attn':
        model_component = model.transformer.h[layer].attn
        model_component_str = f'model.transformer.h[{layer}].attn'
    elif component == 'mlp':
        model_component = model.transformer.h[layer].mlp
        model_component_str = f'model.transformer.h[{layer}].mlp'
    elif component == 'block':
        model_component = model.transformer.h[layer]
        model_component_str = f'model.transformer.h[{layer}]'
    
    results_matrix = []
    
    for probe_symbol in tqdm(vocab_symbols, desc="Evaluating generalization"):
        with torch.no_grad():
            batch = sample_balanced_batch_for_symbol(
                task, batch_size, probe_symbol, num_shots, probe_type=probe_type
            )
            # Remove token_idx parameter
            batch_act_cache = cache_activations(
                model, batch['prompts'], model_component, model_component_str
            )
            
            torch.cuda.empty_cache()
            
            # Select last token manually
            data_X = batch_act_cache[model_component_str][:, -1, :] @ subspace_proj
            data_Y = batch['probe_labels'].long()
            
            # Test all probes on this data
            row_accs = []
            for v in vocab_symbols:
                probe = probe_results[v]['probe']
                probe.eval()
                with torch.no_grad():
                    outputs = probe(data_X).squeeze()
                    preds = (outputs > 0.5).float()
                    acc = (preds == data_Y).float().mean().item()
                row_accs.append(acc)
            
            results_matrix.append(row_accs)
        
        torch.cuda.empty_cache()
    
    return torch.Tensor(results_matrix)

