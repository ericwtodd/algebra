import random
import torch
from .group_utils import determine_associative_pairs

### Simple Versions of Algorithmic Verification
def check_copyable(sequence):
    """
    Determines whether copying could solve the sequence.
    
    Args:
        sequence (str): Comma-separated sequence of facts ending with a query fact.
                       Format: ",ab=c,cd=e,...,xy=z"
    
    Returns:
        bool: True if the query fact has already appeared in the sequence, False otherwise.
    """
    facts = sequence.split(',')
    query = facts[-1]
    return any([fact.split('=')[0] == query.split('=')[0] for fact in facts[:-1]])

def check_reverse_copyable(sequence):
    """
    Determines whether commutative copying could solve the sequence.
    
    Args:
        sequence (str): Comma-separated sequence of facts ending with a query fact.
                       Format: ",ab=c,cd=e,...,xy=z"
    
    Returns:
        bool: True if the reversed query fact has already appeared in the sequence,
              False otherwise.
    """
    facts = sequence.split(',')
    query = facts[-1]
    return any([fact.split('=')[0] == query.split('=')[0][::-1] and fact.split('=')[1] == query.split('=')[1] for fact in facts[1:-1]])


def check_identity(sequence):
    """
    Determines whether identity recognition could solve the sequence.
    
    Checks if any fact in the sequence (excluding the first and last) has matching
    left and right sides, and if either side appears in the query.
    
    Args:
        sequence (str): Comma-separated sequence of facts ending with a query fact.
                       Format: ",ab=c,cd=e,...,xy=z"
    
    Returns:
        bool: True if identity recognition could solve the query, False otherwise.
    """
    facts = sequence.split(',')
    query = facts[-1]

    left_identity = [fact[0] == fact[-1] and fact[1] in query.split('=')[0] for fact in facts[1:-1]]
    right_identity = [fact[1] == fact[-1] and fact[0] in query.split('=')[0] for fact in facts[1:-1]]

    return any(left_identity or right_identity)

def check_closure_elimination_solvable(sequence):
    """
    Determines whether a sequence is solvable via closure and cancellation law.
    
    This algorithm checks if the query can be solved by:
    1. Finding facts that share symbols with the query
    2. Computing the closure over all symbols in those facts
    3. Computing the closure over answer symbols from facts with query_a in left slot 
       or query_b in right slot
    4. Checking if the set difference is a singleton (the answer)
    
    Args:
        sequence (str): Comma-separated sequence of facts ending with a query fact.
                       Format: ",fk=i,kn=g,cd=d,kh=c,in=c,nf=h,cg=g,if=n,gf=c,id=h,cg=g,df=g"
    
    Returns:
        bool: True if the sequence is solvable via closure elimination, False otherwise.
    """
    facts = sequence.split(',')
    query = facts[-1]
    
    share_a_on_left = [fact for fact in facts[1:-1] if fact[0] == query[0]]
    share_b_on_right = [fact for fact in facts[1:-1] if fact[1] == query[1]]

    share_symbol_slots = share_a_on_left + share_b_on_right

    def get_closure_set(facts):
        return set(''.join([x for x in facts]).replace('=', ''))

    set_closure = get_closure_set(share_symbol_slots) # includes answers
    answer_closure = get_closure_set([x[-1] for x in share_symbol_slots])
    
    return len(set_closure - answer_closure) == 1 and (set_closure - answer_closure) == set(sequence[-1])

def check_associative(sequence):
    """
    Determines whether associative reasoning could solve the sequence.
    
    Checks if there exists a triplet of facts that can be composed associatively
    to derive the query fact.
    
    Args:
        sequence (str): Comma-separated sequence of facts ending with a query fact.
                       Format: ",fk=i,kn=g,cd=d,kh=c,in=c,nf=h,cg=g,if=n,gf=c,id=h,cg=g,df=g"
    
    Returns:
        bool: True if the query can be solved via associativity, False otherwise.
    """
    facts = sequence.split(',')
    query = facts[-1]

    triplets = determine_associative_pairs(query)

    is_associative=False
    for triplet in triplets:
        all_facts_exist = True
        for fact in triplet:
            if fact not in facts: # Need each fact of an associative triplet
                all_facts_exist=False
                break
        if all_facts_exist: # If there is a triplet of facts that compose to solve the query
            is_associative=True
            break
    return is_associative

### Optimized Torch Versions of Algorithmic Verification
def copy_rates_over_sequence(batch, exclude_masks=None):
    """
    Analyzes what fraction of queries can be solved by direct copying at each position.
    
    For each query position in the sequence, determines if the exact same query 
    (same two-element key) has appeared earlier in the sequence.
    
    Args:
        batch (dict or torch.Tensor): Either a dict with 'inputs' key or a tensor of 
                                      shape [batch_size, seq_len] containing token IDs.
        exclude_masks (dict, optional): Dictionary mapping mask names to boolean tensors 
                                       of shape [batch_size, num_facts+1]. When provided, 
                                       only cases where all masks are False are counted.
    
    Returns:
        tuple: (copy_rates, copy_stds, can_copy_tensor)
            - copy_rates (torch.Tensor): Rate at each position [num_facts+1]
            - copy_stds (torch.Tensor): Standard deviation at each position [num_facts+1]
            - can_copy_tensor (torch.BoolTensor): Boolean mask [batch_size, num_facts+1]
    """
    # Handle both dict and tensor inputs
    if isinstance(batch, dict):
        batch = batch['inputs']
    
    batch_size, seq_len = batch.shape
    
    # Parse sequence structure: facts are 5 tokens each (comma, a, b, =, c)
    num_complete_facts = (seq_len - 4) // 5
    complete_tokens = num_complete_facts * 5
    facts = batch[:, :complete_tokens].view(batch_size, num_complete_facts, 5)
    
    # Get all query keys (including final incomplete query)
    complete_query_keys = facts[:, :, 1:3]
    last_query_keys = batch[:, complete_tokens+1:complete_tokens+3]
    all_query_keys = torch.cat([complete_query_keys, last_query_keys.unsqueeze(1)], dim=1)
    
    copy_rates = [0.0]  # First position can't copy
    copy_stds = [0.0]
    can_copy_tensor = torch.zeros(batch_size, num_complete_facts + 1, dtype=torch.bool)
    
    for pos in range(1, num_complete_facts + 1):
        current_queries = all_query_keys[:, pos]
        previous_queries = all_query_keys[:, :pos]
        
        # Check if current query matches any previous query
        matches = torch.all(previous_queries == current_queries.unsqueeze(1), dim=2)
        can_copy = torch.any(matches, dim=1)
        can_copy_tensor[:, pos] = can_copy
        
        # Apply exclusion filters
        if exclude_masks:
            valid_cases = torch.ones(batch_size, dtype=torch.bool)
            for mask_tensor in exclude_masks.values():
                if mask_tensor is not None:
                    valid_cases = valid_cases & (~mask_tensor[:, pos])
            copy_and_valid = can_copy & valid_cases
            rate = copy_and_valid.float().mean().item()
            std = copy_and_valid.float().std().item()
        else:
            rate = can_copy.float().mean().item()
            std = can_copy.float().std().item()
        
        copy_rates.append(rate)
        copy_stds.append(std)
    
    return torch.tensor(copy_rates), torch.tensor(copy_stds), can_copy_tensor


def reverse_copy_rates_over_sequence(batch, exclude_masks=None):
    """
    Analyzes what fraction of queries can be solved by copying the reverse (commutative) pair.
    
    For each query (a, b), checks if the reversed pair (b, a) has appeared earlier 
    in the sequence.
    
    Args:
        batch (dict or torch.Tensor): Either a dict with 'inputs' key or a tensor of 
                                      shape [batch_size, seq_len] containing token IDs.
        exclude_masks (dict, optional): Dictionary mapping mask names to boolean tensors 
                                       of shape [batch_size, num_facts+1]. When provided, 
                                       only cases where all masks are False are counted.
    
    Returns:
        tuple: (reverse_copy_rates, reverse_copy_stds, can_reverse_copy_tensor)
            - reverse_copy_rates (torch.Tensor): Rate at each position [num_facts+1]
            - reverse_copy_stds (torch.Tensor): Standard deviation at each position [num_facts+1]
            - can_reverse_copy_tensor (torch.BoolTensor): Boolean mask [batch_size, num_facts+1]
    """
    # Handle both dict and tensor inputs
    if isinstance(batch, dict):
        batch = batch['inputs']
    
    batch_size, seq_len = batch.shape
    
    # Parse sequence structure
    num_complete_facts = (seq_len - 4) // 5
    complete_tokens = num_complete_facts * 5
    facts = batch[:, :complete_tokens].view(batch_size, num_complete_facts, 5)
    
    # Get all query keys (including final incomplete query)
    complete_query_keys = facts[:, :, 1:3]
    last_query_keys = batch[:, complete_tokens+1:complete_tokens+3]
    all_query_keys = torch.cat([complete_query_keys, last_query_keys.unsqueeze(1)], dim=1)
    
    reverse_copy_rates = [0.0]  # First position can't copy
    reverse_copy_stds = [0.0]
    can_reverse_copy_tensor = torch.zeros(batch_size, num_complete_facts + 1, dtype=torch.bool)
    
    for pos in range(1, num_complete_facts + 1):
        current_queries = all_query_keys[:, pos]
        previous_queries = all_query_keys[:, :pos]
        
        # Create reversed version of current queries: (b, a) instead of (a, b)
        reversed_current = torch.stack([current_queries[:, 1], current_queries[:, 0]], dim=1)
        
        # Check if reversed version appears in previous queries
        matches = torch.all(previous_queries == reversed_current.unsqueeze(1), dim=2)
        can_reverse_copy = torch.any(matches, dim=1)
        
        can_reverse_copy_tensor[:, pos] = can_reverse_copy
        
        # Apply exclusion filters
        if exclude_masks:
            valid_cases = torch.ones(batch_size, dtype=torch.bool)
            for mask_tensor in exclude_masks.values():
                if mask_tensor is not None:
                    valid_cases = valid_cases & (~mask_tensor[:, pos])
            reverse_and_valid = can_reverse_copy & valid_cases
            rate = reverse_and_valid.float().mean().item()
            std = reverse_and_valid.float().std().item()
        else:
            rate = can_reverse_copy.float().mean().item()
            std = can_reverse_copy.float().std().item()
        
        reverse_copy_rates.append(rate)
        reverse_copy_stds.append(std)
    
    return torch.tensor(reverse_copy_rates), torch.tensor(reverse_copy_stds), can_reverse_copy_tensor


def identity_rates_over_sequence(batch, exclude_masks=None):
    """
    Analyzes what fraction of queries can be solved by the identity rule.
    
    The identity rule checks if any previous fact has the form x=x or y=y, where
    x or y appears in the current query. If so, that element can be treated as 
    an identity element.
    
    Args:
        batch (dict or torch.Tensor): Either a dict with 'inputs' key or a tensor of 
                                      shape [batch_size, seq_len] containing token IDs.
        exclude_masks (dict, optional): Dictionary mapping mask names to boolean tensors 
                                       of shape [batch_size, num_facts+1]. When provided, 
                                       only cases where all masks are False are counted.
    
    Returns:
        tuple: (identity_rates, identity_stds, can_identity_tensor)
            - identity_rates (torch.Tensor): Rate at each position [num_facts+1]
            - identity_stds (torch.Tensor): Standard deviation at each position [num_facts+1]
            - can_identity_tensor (torch.BoolTensor): Boolean mask [batch_size, num_facts+1]
    """
    # Handle both dict and tensor inputs
    if isinstance(batch, dict):
        batch = batch['inputs']
    
    batch_size, seq_len = batch.shape
    
    # Parse sequence structure
    num_complete_facts = (seq_len - 4) // 5
    complete_tokens = num_complete_facts * 5
    facts = batch[:, :complete_tokens].view(batch_size, num_complete_facts, 5)
    
    # Get all query keys
    complete_query_keys = facts[:, :, 1:3]
    last_query_keys = batch[:, complete_tokens+1:complete_tokens+3]
    all_query_keys = torch.cat([complete_query_keys, last_query_keys.unsqueeze(1)], dim=1)
    
    identity_rates = [0.0]
    identity_stds = [0.0]
    can_identity_tensor = torch.zeros(batch_size, num_complete_facts + 1, dtype=torch.bool)
    
    for pos in range(1, num_complete_facts + 1):
        current_queries = all_query_keys[:, pos]
        previous_facts = facts[:, :pos]
        
        # Vectorized identity detection
        prev_keys = previous_facts[:, :, 1:3]
        prev_answers = previous_facts[:, :, 4]
        x_is_identity = prev_keys[:, :, 0] == prev_answers
        y_is_identity = prev_keys[:, :, 1] == prev_answers
        
        can_solve_identity = torch.zeros(batch_size, dtype=torch.bool)
        
        for batch_idx in range(batch_size):
            query = current_queries[batch_idx]
            # Get all identity elements from previous facts
            x_ids = prev_keys[batch_idx, x_is_identity[batch_idx], 1]
            y_ids = prev_keys[batch_idx, y_is_identity[batch_idx], 0]
            all_ids = torch.cat([x_ids, y_ids]) if len(x_ids) > 0 or len(y_ids) > 0 else torch.tensor([])
            
            # Check if any query element matches an identity element
            if len(all_ids) > 0:
                can_solve_identity[batch_idx] = (query[0].unsqueeze(0) == all_ids).any() or (query[1].unsqueeze(0) == all_ids).any()
        
        can_identity_tensor[:, pos] = can_solve_identity
        
        # Apply exclusion filters
        if exclude_masks:
            valid_cases = torch.ones(batch_size, dtype=torch.bool)
            for mask_tensor in exclude_masks.values():
                if mask_tensor is not None:
                    valid_cases = valid_cases & (~mask_tensor[:, pos])
            identity_and_valid = can_solve_identity & valid_cases
            rate = identity_and_valid.float().mean().item()
            std = identity_and_valid.float().std().item()
        else:
            rate = can_solve_identity.float().mean().item()
            std = can_solve_identity.float().std().item()
            
        identity_rates.append(rate)
        identity_stds.append(std)
    
    return torch.tensor(identity_rates), torch.tensor(identity_stds), can_identity_tensor


def closure_elimination_rates_over_sequence(batch, eliminate_copies=False, exclude_masks=None):
    """
    Analyzes what fraction of queries can be solved by closure+elimination rule.
    
    The closure+elimination algorithm:
    1. Finds facts that share symbols with the query (optionally excluding exact/commutative copies)
    2. Computes closure over all symbols in these facts
    3. Computes closure over answer symbols from facts with query_a in left slot or query_b in right slot
    4. Checks if set difference is a singleton (the answer)
    
    Args:
        batch (dict or torch.Tensor): Either a dict with 'inputs' key or a tensor of 
                                      shape [batch_size, seq_len] containing token IDs.
        eliminate_copies (bool): If True, excludes exact and commutative copies from 
                                closure computation.
        exclude_masks (dict, optional): Dictionary mapping mask names to boolean tensors 
                                       of shape [batch_size, num_facts+1]. When provided, 
                                       only cases where all masks are False are counted.
    
    Returns:
        tuple: (closure_rates, closure_stds, can_closure_tensor)
            - closure_rates (torch.Tensor): Rate at each position [num_facts+1]
            - closure_stds (torch.Tensor): Standard deviation at each position [num_facts+1]
            - can_closure_tensor (torch.BoolTensor): Boolean mask [batch_size, num_facts+1]
    """
    # Handle both dict and tensor inputs
    if isinstance(batch, dict):
        batch = batch['inputs']
    
    batch_size, seq_len = batch.shape
    
    # Parse sequence structure
    num_complete_facts = (seq_len - 4) // 5
    complete_tokens = num_complete_facts * 5
    facts = batch[:, :complete_tokens].view(batch_size, num_complete_facts, 5)
    
    # Get all query keys
    complete_query_keys = facts[:, :, 1:3]
    last_query_keys = batch[:, complete_tokens+1:complete_tokens+3]
    all_query_keys = torch.cat([complete_query_keys, last_query_keys.unsqueeze(1)], dim=1)
    
    closure_rates = [0.0]  # First position can't use closure+elimination
    closure_stds = [0.0]
    can_closure_tensor = torch.zeros(batch_size, num_complete_facts + 1, dtype=torch.bool)
    
    for pos in range(1, num_complete_facts + 1):
        current_queries = all_query_keys[:, pos]
        previous_facts = facts[:, :pos]
        
        can_solve_closure = torch.zeros(batch_size, dtype=torch.bool)
        
        for batch_idx in range(batch_size):
            query_a, query_b = current_queries[batch_idx]
            prev_facts_batch = previous_facts[batch_idx]
            prev_keys = prev_facts_batch[:, 1:3]
            
            # Find facts that share symbols with query (anywhere in the fact)
            fact_symbols = prev_facts_batch[:, [1, 2, 4]]
            shares_a = (fact_symbols == query_a).any(dim=1)
            shares_b = (fact_symbols == query_b).any(dim=1)
            
            # Optionally exclude exact and commutative copies
            exact_match = (prev_keys[:, 0] == query_a) & (prev_keys[:, 1] == query_b)
            comm_match = (prev_keys[:, 0] == query_b) & (prev_keys[:, 1] == query_a)
            is_copy = exact_match | comm_match
            
            if eliminate_copies:
                # Shares symbols with query AND is not a copy
                shares_symbol = (shares_a | shares_b) & ~is_copy
            else:
                shares_symbol = (shares_a | shares_b)
            
            if not shares_symbol.any():
                continue
                
            # Get the relevant facts (share symbols but aren't copies)
            relevant_facts = prev_facts_batch[shares_symbol]
            
            # Get closure over all symbols in relevant facts
            relevant_symbols = relevant_facts[:, [1, 2, 4]]
            set_closure = torch.unique(relevant_symbols.flatten())
            
            # Get closure over answer symbols from facts that have query_a in left slot OR query_b in right slot
            if eliminate_copies:
                non_copy_facts = prev_facts_batch[~is_copy]
                a_in_left = non_copy_facts[:, 1] == query_a
                b_in_right = non_copy_facts[:, 2] == query_b
            else:
                a_in_left = prev_facts_batch[:, 1] == query_a
                b_in_right = prev_facts_batch[:, 2] == query_b
            answer_relevant = a_in_left | b_in_right
            
            if answer_relevant.any():
                if eliminate_copies:
                    answer_facts = non_copy_facts[answer_relevant]
                else:
                    answer_facts = prev_facts_batch[answer_relevant]
                answer_closure = torch.unique(answer_facts[:, 4])
                
                # Set difference
                set_closure_set = set(set_closure.tolist())
                answer_closure_set = set(answer_closure.tolist())
                difference = set_closure_set - answer_closure_set
                
                # Check if difference is singleton
                can_solve_closure[batch_idx] = len(difference) == 1
        
        can_closure_tensor[:, pos] = can_solve_closure
        
        # Apply exclusion filters
        if exclude_masks:
            valid_cases = torch.ones(batch_size, dtype=torch.bool)
            for mask_tensor in exclude_masks.values():
                if mask_tensor is not None:
                    valid_cases = valid_cases & (~mask_tensor[:, pos])
            closure_and_valid = can_solve_closure & valid_cases
            rate = closure_and_valid.float().mean().item()
            std = closure_and_valid.float().std().item()
        else:
            rate = can_solve_closure.float().mean().item()
            std = can_solve_closure.float().std().item()
        
        closure_rates.append(rate)
        closure_stds.append(std)
    
    return torch.tensor(closure_rates), torch.tensor(closure_stds), can_closure_tensor


def random_seen_guess_rates_over_sequence(batch, exclude_masks=None):
    """
    Analyzes what fraction of queries can be solved by random guessing from seen tokens.
    
    For each query position, calculates the probability of correctly guessing the answer
    by randomly selecting from all element tokens (a, b, c values) seen so far in the 
    sequence, including the current query's input tokens.
    
    Args:
        batch (dict): Dictionary containing 'inputs' and 'targets' tensors, both of shape
                     [batch_size, seq_len].
        exclude_masks (dict, optional): Dictionary mapping mask names to boolean tensors 
                                       of shape [batch_size, num_facts+1]. When provided, 
                                       only cases where all masks are False are counted.
    
    Returns:
        tuple: (guess_rates, guess_stds, can_guess_tensor)
            - guess_rates (torch.Tensor): Expected probability at each position [num_facts+1]
            - guess_stds (torch.Tensor): Standard deviation at each position [num_facts+1]
            - can_guess_tensor (torch.FloatTensor): Probability values [batch_size, num_facts+1]
    
    Raises:
        ValueError: If batch is not a dictionary with 'inputs' and 'targets' keys.
    """
    # Handle both dict and tensor inputs
    if isinstance(batch, dict):
        inputs = batch['inputs']
        targets = batch['targets']
    else:
        raise ValueError("batch must be a dictionary with 'inputs' and 'targets' keys for random_seen_guess_rates_over_sequence")
    
    batch_size, seq_len = inputs.shape
    
    # Parse sequence structure
    num_complete_facts = (seq_len - 4) // 5
    complete_tokens = num_complete_facts * 5
    facts = inputs[:, :complete_tokens].view(batch_size, num_complete_facts, 5)
    
    # Get all answers (c values) - INCLUDING the final answer from targets
    complete_answers = facts[:, :, 4]  # [batch, num_complete_facts]
    # The final answer is at position complete_tokens + 3 in the targets
    last_answer = targets[:, complete_tokens + 3]  # [batch] - answer for incomplete query
    all_answers = torch.cat([complete_answers, last_answer.unsqueeze(1)], dim=1)  # [batch, num_facts+1]
    
    guess_rates = []
    guess_stds = []
    can_guess_tensor = torch.zeros(batch_size, num_complete_facts + 1, dtype=torch.float32)
    
    for pos in range(num_complete_facts + 1):       
        # Compute probability of correct guess for each sample
        guess_probs = torch.zeros(batch_size, dtype=torch.float32)
        
        for batch_idx in range(batch_size):
            # For all positions (including the final one), use facts up to but not including current position
            prev_facts = facts[batch_idx, :pos]
            
            # Extract element tokens (a, b, c from each fact) and current query tokens
            seen_tokens = prev_facts[:, [1, 2, 4]].flatten()
            if pos < num_complete_facts:
                current_query_tokens = facts[batch_idx, pos][[1, 2]].flatten()
            else:
                current_query_tokens = inputs[batch_idx, -4:][[1, 2]].flatten()
            seen_tokens = torch.cat([seen_tokens, current_query_tokens])
            seen_tokens = torch.unique(seen_tokens)
            
            # Get the current answer
            current_answer = all_answers[batch_idx, pos]
            
            # Probability of correct guess = 1/|seen_tokens| if answer is in seen_tokens, else 0
            if current_answer in seen_tokens:
                guess_probs[batch_idx] = 1.0 / len(seen_tokens)
            else:
                guess_probs[batch_idx] = 0.0
        
        can_guess_tensor[:, pos] = guess_probs
        
        # Apply exclusion filters if provided
        if exclude_masks:
            valid_cases = torch.ones(batch_size, dtype=torch.bool)
            for mask_tensor in exclude_masks.values():
                if mask_tensor is not None:
                    valid_cases = valid_cases & (~mask_tensor[:, pos])
            
            # Calculate rate only for valid cases
            valid_probs = guess_probs * valid_cases.float()
            rate = valid_probs.mean().item()
            std = valid_probs.std().item()
        else:
            # All cases
            rate = guess_probs.mean().item()
            std = guess_probs.std().item()
        
        guess_rates.append(rate)
        guess_stds.append(std)
    
    return torch.tensor(guess_rates), torch.tensor(guess_stds), can_guess_tensor


def associativity_rates_over_sequence(batch_dict, task, min_3_facts=False, exclude_masks=None):
    """
    Analyzes what fraction of queries can be solved by associativity rule.
    
    The associativity rule checks if there exists a triplet of facts that can be 
    composed associatively to derive the query fact. For a query (a, b), it looks
    for triplets like:
    - (a*g=f, g*d=b, f*d=c) -> ab=c, or
    - (d*b=f, g*d=a, g*f=c) -> ab=c
    
    Args:
        batch_dict (dict): Dictionary containing:
            - 'inputs' (torch.Tensor): Input sequences [batch_size, seq_len]
            - 'groups' (tuple): Tuple of group objects for each batch sample
            - 'orders' (tuple): Tuple of group orders for each batch sample
            - 'vocabulary' (list): List of vocabulary strings for each batch sample
        task: Task object (needed for task.vocab and task.numfor to map between 
              token IDs and characters).
        min_3_facts (bool): If True, only considers triplets with exactly 3 unique facts.
                           If False, allows triplets with duplicate facts.
        exclude_masks (dict, optional): Dictionary mapping mask names to boolean tensors 
                                       of shape [batch_size, num_facts+1]. When provided, 
                                       only cases where all masks are False are counted.
    
    Returns:
        tuple: (assoc_rates, assoc_stds, can_assoc_tensor)
            - assoc_rates (torch.Tensor): Rate at each position [num_facts+1]
            - assoc_stds (torch.Tensor): Standard deviation at each position [num_facts+1]
            - can_assoc_tensor (torch.BoolTensor): Boolean mask [batch_size, num_facts+1]
    
    Raises:
        ValueError: If batch_dict is not a dictionary with required keys.
    """
    # Handle both dict and tensor inputs
    if isinstance(batch_dict, dict):
        batch = batch_dict['inputs']
        groups_list = batch_dict['groups']
        orders_list = batch_dict['orders']
        vocab_list = batch_dict['vocabulary']
    else:
        raise ValueError("batch_dict must be a dictionary with 'inputs', 'groups', 'orders', 'vocabulary' keys")
    
    batch_size, seq_len = batch.shape
    num_complete_facts = (seq_len - 4) // 5
    complete_tokens = num_complete_facts * 5
    facts = batch[:, :complete_tokens].view(batch_size, num_complete_facts, 5)
    
    # Get all query keys
    complete_query_keys = facts[:, :, 1:3]
    last_query_keys = batch[:, complete_tokens+1:complete_tokens+3]
    all_query_keys = torch.cat([complete_query_keys, last_query_keys.unsqueeze(1)], dim=1)
    
    # PRE-COMPUTATION: Build mappings once per batch sample
    vocab_mappings = []  # token_id -> vocab_position
    cumsum_orders_list = []
    elem_to_pos_list = []  # For each batch: list of dicts, one per group
    
    for batch_idx in range(batch_size):
        vocab = vocab_list[batch_idx]
        orders = orders_list[batch_idx]
        
        # Build token_id -> vocab_position mapping
        token_to_pos = {}
        for pos_idx, char in enumerate(vocab):
            token_id = task.numfor[char]
            token_to_pos[token_id] = pos_idx
        vocab_mappings.append(token_to_pos)
        
        # Pre-compute cumsum for group boundaries
        cumsum_orders = [0] + list(torch.tensor(orders).cumsum(0).tolist())
        cumsum_orders_list.append(cumsum_orders)
        
        # Pre-compute element -> vocab position mapping for each group
        group_mappings = []
        for group_idx, order in enumerate(orders):
            group_start = cumsum_orders[group_idx]
            elem_to_pos = {i: group_start + i for i in range(order)}
            group_mappings.append(elem_to_pos)
        elem_to_pos_list.append(group_mappings)
    
    # INCREMENTAL FACT TRACKING: Build prev_fact_pairs incrementally
    prev_fact_pairs_list = [set() for _ in range(batch_size)]
    
    assoc_rates = [0.0]
    assoc_stds = [0.0]
    can_assoc_tensor = torch.zeros(batch_size, num_complete_facts + 1, dtype=torch.bool)
    
    for pos in range(1, num_complete_facts + 1):
        # Add facts from position pos-1 to each batch's previous fact set
        if pos > 1:
            for batch_idx in range(batch_size):
                fact = facts[batch_idx, pos-1]
                a_tok, b_tok = fact[1].item(), fact[2].item()
                vocab_map = vocab_mappings[batch_idx]
                if a_tok in vocab_map and b_tok in vocab_map:
                    prev_fact_pairs_list[batch_idx].add((vocab_map[a_tok], vocab_map[b_tok]))
        
        can_solve_assoc = torch.zeros(batch_size, dtype=torch.bool)
        
        for batch_idx in range(batch_size):
            query_tokens = all_query_keys[batch_idx, pos]
            query_a_token, query_b_token = query_tokens[0].item(), query_tokens[1].item()
            
            vocab_map = vocab_mappings[batch_idx]
            
            # Quick lookup using pre-built mapping
            if query_a_token not in vocab_map or query_b_token not in vocab_map:
                continue
            
            pos_a = vocab_map[query_a_token]
            pos_b = vocab_map[query_b_token]
            
            # Determine groups using pre-computed cumsum
            cumsum_orders = cumsum_orders_list[batch_idx]
            orders = orders_list[batch_idx]
            
            group_a = next(i for i in range(len(orders)) if pos_a < cumsum_orders[i+1])
            group_b = next(i for i in range(len(orders)) if pos_b < cumsum_orders[i+1])
            
            # Query must be within the same group
            if group_a != group_b:
                continue
            
            # Get element indices within the group
            elem_a = pos_a - cumsum_orders[group_a]
            elem_b = pos_b - cumsum_orders[group_b]
            
            # Get associative triplets for this query
            group = groups_list[batch_idx][group_a]
            triplets = determine_associative_pairs(
                (elem_a, elem_b),
                group,
                drop_X=True,
                drop_R=True,
                drop_duplicates=True
            )
            
            if min_3_facts:
                triplets = [t for t in triplets if len(t) == 3]
            
            if not triplets:
                continue
            
            # Use pre-computed element -> position mapping
            elem_to_pos = elem_to_pos_list[batch_idx][group_a]
            
            # Use incrementally-built previous facts set
            prev_fact_pairs = prev_fact_pairs_list[batch_idx]
            
            # Check each triplet to see if all facts exist
            for triplet in triplets:
                all_facts_exist = True
                for (elem_a_t, elem_b_t) in triplet:
                    pos_a_t = elem_to_pos.get(elem_a_t)
                    pos_b_t = elem_to_pos.get(elem_b_t)
                    
                    if pos_a_t is None or pos_b_t is None:
                        all_facts_exist = False
                        break
                    
                    if (pos_a_t, pos_b_t) not in prev_fact_pairs:
                        all_facts_exist = False
                        break
                
                if all_facts_exist:
                    can_solve_assoc[batch_idx] = True
                    break
        
        can_assoc_tensor[:, pos] = can_solve_assoc
        
        # Apply exclusion filters
        if exclude_masks:
            valid_cases = torch.ones(batch_size, dtype=torch.bool)
            for mask_tensor in exclude_masks.values():
                if mask_tensor is not None:
                    valid_cases = valid_cases & (~mask_tensor[:, pos])
            assoc_and_valid = can_solve_assoc & valid_cases
            rate = assoc_and_valid.float().mean().item()
            std = assoc_and_valid.float().std().item()
        else:
            rate = can_solve_assoc.float().mean().item()
            std = can_solve_assoc.float().std().item()
        
        assoc_rates.append(rate)
        assoc_stds.append(std)
    
    return torch.tensor(assoc_rates), torch.tensor(assoc_stds), can_assoc_tensor