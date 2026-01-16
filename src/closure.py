import torch
import numpy as np
import random
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import nnsight

from .intervention_utils import cache_activations, inference_w_subspace_patch
from .group_utils import get_pair_with_answer
from .load_utils import load_gpt, load_task

### Generate Closure Counterfactuals 

def find_closure_intervention_counterfactual(CS, CB, QS, QB, intervention_type):
    """
    Find valid coset configurations that satisfy specific counterfactual constraints.
    
    This function determines how to partition closure symbols into left (L) and right (R)
    cosets such that specific elements are excluded under intervention, creating counterfactual scenarios
    for testing model implementation of closure + cancellation.
    
    Args:
        CS (list): Source closure - symbols available in the source prompt
        CB (list): Base closure - symbols available in the base prompt
        QS (str): Source query answer - the symbol that answers the source query
        QB (str): Base query answer - the symbol that answers the base query
        intervention_type (str): Which set to intervene on from source to base:
            - "C": Closure set is replaced with the source closure set
            - "L": Left cancellation coset is replaced with source
            - "R": Right cancellation coset  is replaced with source
            - "LUR": Entire (left and right) cancellation coset is replaced with source cancellation set
    
    Returns:
        tuple or None: (LS, LB, RS, RB, [cf]) where:
            - LS: Left coset symbols from source closure
            - LB: Left coset symbols from base closure
            - RS: Right coset symbols from source closure
            - RB: Right coset symbols from base closure
            - cf: The counterfactual symbol
        Returns None if no valid solution exists.
    
    Constraints Enforced:
        Base constraints (always be solvable via closure + cancellation):
        1. set(CS) - (LS U RS) = {QS}  (source closure set - source cancellation set must equal source answer)
        2. set(CB) - (LB U RB) = {QB}  (base closure set - base cancellation set must equal source answer)
        
        Type-specific constraints:
        - "C": set(CS) - (LB U RB) = {cf}
        - "L": set(CB) - (LS U RB) = {cf}
        - "R": set(CB) - (LB U RS) = {cf}
        - "LUR": set(CB) - (LS U RS) = {cf}
    """  
    # Determine the appropriate counterfactual element pool based on constraint type
    if intervention_type in ["LUR", "L", "R"]:
        cf_pool = set(CB) - set(QB)  # cf comes from CB
    elif intervention_type == "C":
        cf_pool = set(CS) - set(QS)  # cf can be any element in CS except qs
    else:
        raise ValueError(f"Unknown intervention type: {intervention_type}")
    
    valid_solutions = []

    # Try each potential counterfactual value
    for cf in cf_pool:
        # Initialize L and R sets
        LS = set()
        LB = set()
        RS = set()
        RB = set()
        
        # Process CS elements
        for x in CS:
            if x == QS:
                continue  # Skip qs to satisfy base constraint 1
            
            # Add all other CS elements to both LS and RS for all constraint types
            LS.add(x)
            RS.add(x)
        
        # Process CB elements
        for x in CB:
            if x == QB:
                continue  # Skip qb to satisfy base constraint 2
            # Handle various constraint types
            if intervention_type == "C" and x == cf:
                # For "C": Exclude cf from LB and RB
                continue
            elif intervention_type == "R" and x == cf:
                # For "R": Add cf only to RB
                RB.add(x)
            elif intervention_type == "L" and x == cf:
                # For "L": Add cf only to LB
                LB.add(x)
            else:
                # For all other elements, add to both LB and RB
                LB.add(x)
                RB.add(x)
        
        # Verify the base constraints
        constraint1 = set(CS) - (LS | RS) == set(QS)
        constraint2 = set(CB) - (LB | RB) == set(QB)
        
        # Verify the specific third constraint based on type
        if intervention_type == "LUR":
            constraint3 = set(CB) - (LS | RS) == set(cf)
        elif intervention_type == "C":
            constraint3 = set(CS) - (LB | RB) == set(cf)
        elif intervention_type == "L":
            constraint3 = set(CB) - (LS | RB) == set(cf)
        elif intervention_type == "R":
            constraint3 = set(CB) - (LB | RS) == set(cf)
                
        # Additional validation constraints
        subset_constraints = (
            LS.issubset(set(CS)) and
            RS.issubset(set(CS)) and
            LB.issubset(set(CB)) and
            RB.issubset(set(CB))
        )
        
        # Return solution if all constraints are satisfied
        if constraint1 and constraint2 and constraint3 and subset_constraints:
            valid_solutions.append((list(LS), list(LB), list(RS), list(RB), [cf]))
    
    if valid_solutions:
        return random.choice(valid_solutions)
    else:
        return None  # No solution found

def generate_constrained_closure_variables(vocab, order, num_diff_symbols=1, intervention_type='C'):
    """
    Generate source and base closure sets with appropriate query symbols based on the constraints.
    
    Creates two overlapping symbol sets (source and base) that share most symbols
    but differ in specific ways depending on the constraint type.
    
    Args:
        vocab (list): Available symbols to choose from
        order (int): Size of each closure set (group order)
        num_diff_symbols (int): Number of symbols that differ between sets.
            Note: Automatically set to 1 for "C" and "LUR" constraint types
        intervention_type (str): Type of constraint to apply ("C", "L", "R", or "LUR")
    
    Returns:
        tuple : (CS, CB, QS, QB) - source closure, base closure, source query, base query
    """
    # Validate num_diff_symbols based on constraint type
    if intervention_type in ["C", "LUR"]:
        if num_diff_symbols != 1:
            print(f"Warning: intervention type '{intervention_type}' requires num_diff_symbols=1")
        num_diff_symbols = 1
    else:
        assert 1 <= num_diff_symbols <= 6, "num_diff_symbols must be between 1 and 6"
    
    # Sample shared and unique symbols
    shared = random.sample(vocab, k=order - num_diff_symbols)
    remaining = list(set(vocab) - set(shared))
    extra_1 = random.sample(remaining, k=num_diff_symbols)
    extra_2 = random.sample(list(set(remaining) - set(extra_1)), k=num_diff_symbols)
    
    # Construct variable sets
    CS = shared + extra_1
    CB = shared + extra_2
    
    # Pick QS and QB based on constraint type
    if intervention_type == "C":
        QS = random.choice(shared)
        QB = random.choice(extra_2)
    elif intervention_type in ["L", "R", "LUR"]:
        QS = random.choice(extra_1)
        QB = random.choice(shared)
    
    random.shuffle(CS)
    random.shuffle(CB)
        
    return CS, CB, QS, QB

def get_coset_facts(group, holdout_pair, reverse_out=True, include_left=True, include_right=True):
    """
    """
    # Create all possible pairs for the Cayley table
    elems = group.elements
    all_pairs = [(a, b) for a in elems for b in elems]
    
    # Remove the holdout pair from available pairs
    available_pairs = all_pairs.copy()
    available_pairs.remove(holdout_pair)
        
    # Optionally remove the reverse pair
    reverse_pair = (holdout_pair[1], holdout_pair[0])
    if reverse_out and reverse_pair in available_pairs:    
        available_pairs.remove(reverse_pair)
    
    facts = {'holdout_pair':[holdout_pair],'available_pairs':available_pairs,'left_coset':[],'right_coset':[]}

    if include_left:
        # Left cosets: all pairs with first element = holdout's first element
        left_coset = [(a, b) for a, b in available_pairs if a == holdout_pair[0]]
        facts['left_coset'] = left_coset

    if include_right:
        # Right cosets: all pairs with second element = holdout's second element
        right_coset = [(a, b) for a, b in available_pairs if b == holdout_pair[1]]
        facts['right_coset'] = right_coset

    return facts

def construct_prompt(facts, holdout_pair, wordfor, num_shots):
    """
    """
    include_set = facts.copy()
        
    for i in range(num_shots - (len(facts))):
        random_fact = random.choice(facts)
        include_set.extend([random_fact])

    random.shuffle(include_set)

    formatted_facts = []
    for pair in include_set + [holdout_pair]:
        formatted_facts.append(f"{wordfor[pair[0]]}{wordfor[pair[1]]}={wordfor[pair[0] * pair[1]]}")
    
    return ',' + ','.join(formatted_facts)

def counterfactual_prompt_pair(task, group=None, num_shots=None, num_diff_symbols=1, intervention_type='C', allow_identity_facts=False, allow_cf_identity_base_facts=True):
   """
   """
   if group is None:
       group = task.sample_groups()[0] # Sample a single group
   order = group.order()
   
   while True:
        variable_set_S, variable_set_B, query_S, query_B = generate_constrained_closure_variables(task.vocab[:task.num_symbols], order, num_diff_symbols, intervention_type)
        left_coset_answers_S, left_coset_answers_B, right_coset_answers_S, right_coset_answers_B, counterfactual_target = find_closure_intervention_counterfactual(variable_set_S, variable_set_B, query_S, query_B, intervention_type)

        CQLR_S = (variable_set_S, query_S, left_coset_answers_S, right_coset_answers_S)
        CQLR_B = (variable_set_B, query_B, left_coset_answers_B, right_coset_answers_B)

        for i, (C,Q,L,R) in enumerate([CQLR_S, CQLR_B]):
            holdout_pair, wordfor, elemfor = get_pair_with_answer(group, C, Q, allow_identity_facts=allow_identity_facts)
            facts = get_coset_facts(group, holdout_pair, reverse_out=True, include_left=True, include_right=True)

            # Filter Cosets
            filtered_left = [(x,y) for x,y in facts['left_coset'] if wordfor[x*y] in L]
            filtered_right = [(x,y) for x,y in facts['right_coset'] if wordfor[x*y] in R]
            fact_list = filtered_left + filtered_right
            if i == 0:
                sentence_dict_S = {'holdout': holdout_pair, 'facts': fact_list, 'wordfor':wordfor, 'vocab':C}
            else:
                sentence_dict_B = {'holdout': holdout_pair, 'facts': fact_list, 'wordfor':wordfor, 'vocab':C}

        if num_shots is None:
            num_shots = max(len(sentence_dict_B['facts']), len(sentence_dict_S['facts']))

        s_S = construct_prompt(sentence_dict_S['facts'], sentence_dict_S['holdout'], sentence_dict_S['wordfor'], num_shots)
        s_B = construct_prompt(sentence_dict_B['facts'], sentence_dict_B['holdout'], sentence_dict_B['wordfor'], num_shots)

        if s_B[-1] != s_S[-1]:
            if allow_cf_identity_base_facts:
                break
            elif counterfactual_target[0] not in s_B[-4:-2]: 
                break

   return s_S, s_B, variable_set_S, variable_set_B, counterfactual_target, order

def sample_batch_counterfactual_prompt_pairs(task, batch_size, group=None, num_shots=None, num_diff_symbols=1, intervention_type='C', allow_identity_facts=False, allow_cf_identity_base_facts=True):
    """
    """

    s1, s2, v1, v2, ct, o = zip(*[counterfactual_prompt_pair(task, group, num_shots=num_shots, num_diff_symbols=num_diff_symbols, intervention_type=intervention_type, allow_identity_facts=allow_identity_facts, allow_cf_identity_base_facts=allow_cf_identity_base_facts) for _ in range(batch_size)])

    s1 = task.tensor_from_expression(s1)
    s2 = task.tensor_from_expression(s2)
    counterfactual_targets = task.tensor_from_expression(ct).squeeze().long()
    # 32 comes from vocab dim in model - can fix hard-code at some point
    # order = group.order()
    # counterfactual_closure_targets = torch.nn.functional.one_hot(torch.Tensor([[task.vocab.index(y) for y in x] for x in v1]).long(), 32).sum(1).float() / o

    inputs1, targets1 = s1[:,:-1], s1[:,-1]
    inputs2, targets2 = s2[:,:-1], s2[:,-1]

    batch = {'source_prompts':inputs1, 'source_targets': targets1, 
             'base_prompts': inputs2, 'base_targets':targets2,
             'source_vocab':v1, 'base_vocab': v2, 
             'counterfactual_targets': counterfactual_targets}
            #  'counterfactual_closure_targets':counterfactual_closure_targets}

    return batch

### Subspace Definition and Optimization

class HouseholderSubspace:
    """
    Construct an orthogonal subspace via Householder reflections.
    
    Args:
        d_embed: model hidden state dimension
        subspace_dim: desired subspace dimension
    """
    def __init__(self, d_embed, subspace_dim=128):        
        self.d_embed = d_embed
        self.subspace_dim = subspace_dim
        
        # Learnable Householder vectors
        self.vectors = torch.nn.Parameter(torch.empty(d_embed, subspace_dim), requires_grad=True)
        torch.nn.init.orthogonal_(self.vectors)
    
    def __repr__(self):
        return f"HouseholderSubspace(d_embed={self.d_embed}, subspace_dim={self.subspace_dim})"
    
    def construct_Q(self):
        """Construct orthogonal matrix Q from Householder reflections."""
        Q = torch.eye(self.d_embed, device=self.vectors.device, dtype=self.vectors.dtype)
        for i in range(self.subspace_dim):
            v = self.vectors[:,i]
            v_norm = v / v.norm()
            Qv = Q @ v_norm
            Q = Q - 2 * torch.outer(Qv, v_norm)
        return Q
    
    def get_subspace_directions(self, k=None):
        """Get the k-dimensional subspace projection matrix Q_k."""
        if k is None:
            k = self.subspace_dim
        Q_full = self.construct_Q()
        return Q_full[:, :k]

def subspace_optim(model, model_component, subspace, train, test, train_source_cache, test_source_cache, loss_fn, optimizer, 
                   mask=None, train_mask=False, intervention_index=-1, n_steps:int=1000, verbose:bool=False, batch_size:int=50, 
                   shuffle:bool=True, seed:int=42, eval_interval:int=2):
    """
    Optimize a mask that picks the best directions to flip predictions on p2 to targets.
    Now supports batched processing of prompts with shuffling for better memory efficiency and training.
    
    Args:
        model: The model to optimize
        model_component: The component to patch
        
        directions: Directions tensor
        train: Dictionary containing 'base_prompts' and 'counterfactual_targets' for training
        test: Dictionary containing 'base_prompts' and 'counterfactual_targets' for evaluation
        train_source_cache: Cache for training activations
        test_source_cache: Cache for test activations
        loss_fn: Loss function
        optimizer: Optimizer
        intervention_index: Index to intervene at (usually -1 (answer-slot) or -3 (left-slot))
        n_steps: Number of optimization steps
        verbose: Whether to print progress
        batch_size: Batch size for processing prompts
        shuffle: Whether to shuffle data between steps
        seed: Random seed for reproducibility
        eval_interval: How often to evaluate on the test set (steps)
    """
    losses = []
    accs = []
    test_accs = []
    best_test_acc = 0
    best_mask = None
        
    n_samples = train['base_prompts'].shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division

    if train_mask:
        assert mask is not None
        mask = mask.to(model.device)
    
    # Set random seed for reproducibility
    if shuffle:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Create indices array for shuffling
    indices = torch.arange(n_samples)
    
    # Create tqdm progress bar
    pbar = tqdm(range(n_steps+1), desc="Subspace Optimization")
    
    # Ensure train and test targets are on the correct device
    # if intervention_index == -3: # this is at the "left-slot"
    #     train_targets = train['counterfactual_closure_targets'].to(model.device)
    #     test_targets = test['counterfactual_closure_targets'].to(model.device)
    # else:
    train_targets = train['counterfactual_targets'].to(model.device)
    test_targets = test['counterfactual_targets'].to(model.device)

    for i in pbar:
        # Shuffle indices at the beginning of each step if shuffle is True
        if shuffle:
            indices = torch.randperm(n_samples)
        
        batch_losses = []
        batch_accs = []
        
        # Process each batch
        for b in range(n_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, n_samples)
            
            # Use shuffled indices to select batch elements
            batch_indices = indices[start_idx:end_idx]
            
            batch_prompts = train['base_prompts'][batch_indices]
            batch_targets = train_targets[batch_indices]
            batch_cache = train_source_cache[batch_indices]
            
            directions = subspace.get_subspace_directions()
            directions = directions.to(model.device)
            
            if train_mask:
                masked_directions = mask * directions
                W_proj = torch.matmul(masked_directions, masked_directions.T)
            else:
                W_proj = torch.matmul(directions, directions.T)

            logits = inference_w_subspace_patch(model, model_component, batch_prompts, W_proj, batch_cache, intervention_index)            
            batch_loss = loss_fn(logits[:,intervention_index], batch_targets)

            if train_mask:
                batch_loss += mask.norm(p=2)
            
            batch_losses.append(batch_loss.item())
            
            if intervention_index == -1:
                batch_accs.append((logits[:,intervention_index].argmax(-1) == batch_targets).float().mean().item())
            # elif intervention_index == -3:
            #     batch_accs.append((logits[:,intervention_idx]))
            else:
                batch_accs.append(0)
            
            # Backprop for this batch               
            batch_loss.backward()
            
            del logits
            torch.cuda.empty_cache()
        
        # Run evaluation on test set at specified intervals
        if i % eval_interval == 0:
            with torch.no_grad():
                if train_mask:
                    masked_directions = mask * directions
                    W_proj_test = torch.matmul(masked_directions, masked_directions.T)
                else:
                    W_proj_test = torch.matmul(directions, directions.T)
                
                test_logits = inference_w_subspace_patch(
                    model, model_component, test['base_prompts'], 
                    W_proj_test, test_source_cache, intervention_index=intervention_index
                )
                if intervention_index == -1:
                    test_acc = (test_logits[:,intervention_index].argmax(-1) == test_targets).float().mean().item()
                    test_accs.append(test_acc)
                else:
                    test_accs.append(0)
                
                if test_accs[-1] > best_test_acc:
                    best_test_acc = test_accs[-1]
                    if train_mask:
                        best_mask = mask.clone()
                
                del test_logits
                torch.cuda.empty_cache()

        # Average loss and accuracy across batches
        avg_loss = sum(batch_losses) / len(batch_losses)
        avg_acc = sum(batch_accs) / len(batch_accs)
        losses.append(avg_loss)
        accs.append(avg_acc)
        
        # Update progress bar with current metrics
        if verbose:
            current_test_acc = test_accs[-1] if test_accs else 0
            pbar.set_postfix({
                "Loss": f"{avg_loss:.4f}", 
                "Train Acc": f"{avg_acc:.4f}", 
                "Val Acc": f"{current_test_acc:.4f}",
                "Approx. Mask Sparsity":f"{mask.sum().item():.4f}" if train_mask else None,
            })
        
        # Update parameters after processing all batches
        optimizer.step()
        optimizer.zero_grad()
        
        
        with torch.no_grad():
            if train_mask:
                mask.clamp_(0,1)

            subspace.vectors /= subspace.vectors.norm(dim=0) # normalize learned directions
            
    results = {
        "losses": losses,
        "train_accs": accs,
        "test_accs": test_accs,
        "best_test_acc": best_test_acc
    }

    if train_mask:
        results.update({'mask':mask, 'best_mask':best_mask if best_mask is not None else mask})
    
    return results


def activations_to_pca_directions(cache, model_component_str):
    """
    """

    # Get activations for the current layer
    acts = cache[model_component_str]

    # Calculate mean for each token position separately
    batch_size, num_positions, hidden_dim = acts.shape
    acts_centered = torch.zeros_like(acts)

    # Center each position independently
    for pos in range(num_positions):
        pos_acts = acts[:, pos, :]  # Shape: [batch_size, hidden_dim]
        pos_mean = pos_acts.mean(dim=0, keepdim=True)  # Mean for this position
        acts_centered[:, pos, :] = pos_acts - pos_mean  # Center this position

    # Reshape to combine all positions for SVD
    acts_centered_flat = acts_centered.reshape(batch_size * num_positions, hidden_dim)

    # Compute SVD on the position-specifically centered data
    U, S, Vh = torch.linalg.svd(acts_centered_flat, full_matrices=False)

    # Convert singular values to eigenvalues
    n_samples = acts_centered_flat.shape[0]
    eigenvalues = (S ** 2) / (n_samples - 1)

    # Columns of V are principal components (rows of Vh) and we want columns to be important directions so we transpose.
    principal_components = Vh.T  
    
    return principal_components

def mask_optim(model, model_component, mask, directions, train, test, train_source_cache, test_source_cache, loss_fn, optimizer, 
               intervention_index=-1, n_steps:int=1000, verbose:bool=False, batch_size:int=50, 
               shuffle:bool=True, seed:int=42, eval_interval:int=2):
    """
    Optimize a mask that picks the best directions (from PCA directions) to flip predictions on p2 to targets.
    Now supports batched processing of prompts with shuffling for better memory efficiency and training.
    
    Args:
        model: The model to optimize
        model_component: The component to patch
        mask: Initial mask to optimize
        directions: Directions tensor
        train: Dictionary containing 'base_prompts' and 'counterfactual_targets' for training
        test: Dictionary containing 'base_prompts' and 'counterfactual_targets' for evaluation
        train_source_cache: Cache for training activations
        test_source_cache: Cache for test activations
        loss_fn: Loss function
        optimizer: Optimizer
        intervention_index: Index to intervene at
        n_steps: Number of optimization steps
        verbose: Whether to print progress
        batch_size: Batch size for processing prompts
        shuffle: Whether to shuffle data between steps
        seed: Random seed for reproducibility
        eval_interval: How often to evaluate on the test set (steps)
    """
    losses = []
    accs = []
    test_accs = []
    best_test_acc = 0
    best_mask = None
    
    n_samples = train['base_prompts'].shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division

    directions = directions.to(model.device)
    mask = mask.to(model.device)
    
    # Set random seed for reproducibility
    if shuffle:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Create indices array for shuffling
    indices = torch.arange(n_samples)
    
    # Create tqdm progress bar
    pbar = tqdm(range(n_steps+1), desc="Mask Optimization")
    
    # Ensure train and test targets are on the correct device
    train_targets = train['counterfactual_targets'].to(model.device)
    test_targets = test['counterfactual_targets'].to(model.device)

    for i in pbar:
        # Shuffle indices at the beginning of each step if shuffle is True
        if shuffle:
            indices = torch.randperm(n_samples)
        
        batch_losses = []
        batch_accs = []
        
        # Process each batch
        for b in range(n_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, n_samples)
            
            # Use shuffled indices to select batch elements
            batch_indices = indices[start_idx:end_idx]
            
            batch_prompts = train['base_prompts'][batch_indices]
            batch_targets = train_targets[batch_indices]
            batch_cache = train_source_cache[batch_indices]
            
            # Apply mask to directions
            masked_directions = mask * directions
            W_proj = torch.matmul(masked_directions, masked_directions.T)

            logits = inference_w_subspace_patch(model, model_component, batch_prompts, W_proj, batch_cache, intervention_index)
                        
            batch_loss = loss_fn(logits[:,intervention_index], batch_targets)
            batch_losses.append(batch_loss.item())
            
            if intervention_index == -1:
                batch_accs.append((logits[:,intervention_index].argmax(-1) == batch_targets).float().mean().item())
            else:
                batch_accs.append(0)
            
            # Backprop for this batch               
            batch_loss.backward()
            
            del logits
            torch.cuda.empty_cache()
        
        # Run evaluation on test set at specified intervals
        if i % eval_interval == 0:
            with torch.no_grad():
                # Apply mask to directions for evaluation
                masked_directions_test = mask * directions
                W_proj_test = torch.matmul(masked_directions_test, masked_directions_test.T)
                
                test_logits = inference_w_subspace_patch(
                    model, model_component, test['base_prompts'], 
                    W_proj_test, test_source_cache, intervention_index=intervention_index
                )
                
                test_acc = (test_logits[:,intervention_index].argmax(-1) == test_targets).float().mean().item()
                test_accs.append(test_acc)
                
                # Save best mask
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_mask = mask.clone()
                
                del test_logits
                torch.cuda.empty_cache()

        # Average loss and accuracy across batches
        avg_loss = sum(batch_losses) / len(batch_losses)
        avg_acc = sum(batch_accs) / len(batch_accs)
        losses.append(avg_loss)
        accs.append(avg_acc)
        
        # Update progress bar with current metrics
        if verbose:
            current_test_acc = test_accs[-1] if test_accs else 0
            pbar.set_postfix({
                "Loss": f"{avg_loss:.4f}", 
                "Train Acc": f"{avg_acc:.4f}", 
                "Val Acc": f"{current_test_acc:.4f}"
            })
        
        # Update parameters after processing all batches
        optimizer.step()
        optimizer.zero_grad()
        
        with torch.no_grad():
            mask.clamp_(0,1)
    
    results = {
        "mask": mask,
        "best_mask": best_mask if best_mask is not None else mask,
        "losses": losses,
        "train_accs": accs,
        "test_accs": test_accs,
        "best_test_acc": best_test_acc
    }
    
    return results

"""
Experiment runner for subspace optimization.
"""

@dataclass
class SubspaceExperimentConfig:
    """All parameters for a single experiment run."""
    # Model
    model_dir: str = '../outputs/mixrosette-facts-10-16-8heads'
    
    # Group (optional - if None, sampled by task)
    group: Optional[Any] = None

    # Component
    component: str = 'attn'   # choices are: ['block', 'attn', 'mlp']
    
    # Intervention
    layer: int = 3
    num_directions: int = 18
    constraint_type: str = 'LUR'
    num_diff_symbols: int = 1
    allow_identity_facts: bool = False
    allow_cf_identity_base_facts: bool = False
    
    # Training
    batch_size: int = 3000
    num_shots: int = 20
    n_steps: int = 50
    train_batch_size: int = 256
    lr: float = 5e-2
    weight_decay: float = 0.01
    train_mask: bool = False
    eval_interval: int = 2
    intervention_index: int = -1


def run_subspace_optimization_experiment(model, task, config: SubspaceExperimentConfig) -> Dict[str, Any]:
    """
    Run a complete subspace optimization experiment.
    
    Returns dict with 'results', 'subspace', and 'config'.
    """
    
    # Setup
    assert config.component in ['block', 'attn', 'mlp']
    if config.component == 'block':
        model_component = model.transformer.h[config.layer]
        model_component_str = f'model.transformer.h[{config.layer}]'
    elif config.component == 'attn':
        model_component = model.transformer.h[config.layer].attn
        model_component_str = f'model.transformer.h[{config.layer}].attn'
    elif config.component == 'mlp':
        model_component = model.transformer.h[config.layer].mlp
        model_component_str = f'model.transformer.h[{config.layer}].mlp'
    
    # Sample data (group sampled automatically by task if not specified)
    train_batch = sample_batch_counterfactual_prompt_pairs(
        task, batch_size=config.batch_size, group=config.group, num_shots=config.num_shots,
        num_diff_symbols=config.num_diff_symbols, intervention_type=config.constraint_type,
        allow_identity_facts=config.allow_identity_facts,
        allow_cf_identity_base_facts=config.allow_cf_identity_base_facts
    )
    test_batch = sample_batch_counterfactual_prompt_pairs(
        task, batch_size=config.batch_size, group=config.group, num_shots=config.num_shots,
        num_diff_symbols=config.num_diff_symbols, intervention_type=config.constraint_type,
        allow_identity_facts=config.allow_identity_facts,
        allow_cf_identity_base_facts=config.allow_cf_identity_base_facts
    )
    
    # Cache activations
    train_cache = cache_activations(model, train_batch['source_prompts'], model_component, model_component_str)
    test_cache = cache_activations(model, test_batch['source_prompts'], model_component, model_component_str)
    
    # Initialize subspace
    HSS = HouseholderSubspace(model.config.n_embd, config.num_directions)
    mask = torch.ones(config.num_directions, requires_grad=True, device=model.device)
    
    train_params = [HSS.vectors]
    if config.train_mask:
        train_params.append(mask)
    
    optimizer = torch.optim.Adam(train_params, lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Train
    results = subspace_optim(
        model, model_component, HSS, train_batch, test_batch,
        train_cache[model_component_str], test_cache[model_component_str],
        loss_fn, optimizer, mask=mask, train_mask=config.train_mask,
        verbose=True, n_steps=config.n_steps, batch_size=config.train_batch_size,
        intervention_index=config.intervention_index, eval_interval=config.eval_interval
    )
    
    # Evaluate
    directions = HSS.get_subspace_directions().to(model.device)
    W_proj = torch.matmul(directions, directions.T)
    
    test_logits = inference_w_subspace_patch(
        model, model_component, test_batch['base_prompts'],
        W_proj, test_cache[model_component_str],
        intervention_index=config.intervention_index
    )
    
    with model.trace(test_batch['base_prompts']):
        orig_logits = model.lm_head.output.save()
    
    idx = config.intervention_index
    orig_acc = (orig_logits[:, idx].argmax(-1) == test_batch['base_targets'].to(model.device)).float().mean().item()
    int_acc = (test_logits[:, idx].argmax(-1) == test_batch['counterfactual_targets'].to(model.device)).float().mean().item()
    
    results['original_accuracy'] = orig_acc
    results['intervention_accuracy'] = int_acc
    
    return {
        'results': results,
        'subspace': HSS.get_subspace_directions().detach().cpu(),
        'config': config
    }


def run_subspace_sweep(model, task, configs: List[SubspaceExperimentConfig]) -> List[Dict[str, Any]]:
    """Run multiple subspace experiments."""
    return [run_subspace_optimization_experiment(model, task, c) for c in configs]