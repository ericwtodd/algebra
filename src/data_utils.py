import random
import torch
from .group_utils import determine_associative_pairs
from .coverage import check_copyable, check_reverse_copyable, check_identity

### Targeted Sequence Construction

def construct_cancellation_sequence(task, group=None, k_shots=None, include_identity=False, unshuffled=False):
    """
    Constructs a sequence that requires closure-based cancellation reasoning to solve.
    
    Creates a sequence of facts from a group where the held-out query fact can be
    derived through cancellation of intermediate elements.
    
    Args:
        task: Task object with vocabulary and sampling methods.
        group: Group object to sample from. If None, samples a random group.
               Can also be a list containing a single group.
        k_shots (int, optional): Number of facts in the sequence. If None, uses all
                                 available facts that share elements with the holdout.
        include_identity (bool): If True, allows identity element in holdout pair.
                                If False, excludes it.
        unshuffled (bool): If True, uses ordered vocabulary. If False, randomly
                          samples vocabulary.
    
    Returns:
        tuple: (sequence_str, [group], [order], [vocab])
            - sequence_str (str): Comma-separated string of facts
            - group (list): List containing the group used
            - order (list): List containing the group order
            - vocab (list): List containing the vocabulary string used
    """
    if group is None:
        group = task.sample_groups()[0]
    elif isinstance(group, list):
        group = group[0]

    # Create all possible pairs from single group
    elems = [(g, 0) for g in group.generate()]
    all_pairs = [(a, b) for a in elems for b in elems]

    # Create vocabulary mapping
    if unshuffled:
        vocab = ''.join(task.vocab[:group.order()])
    else:
        vocab = ''.join(random.sample(task.vocab[:16], k=group.order()))
    wordfor = {g: vocab[i] for i, g in enumerate(elems)}

    # Sample a holdout fact
    while True:
        holdout_pair = random.sample(all_pairs, k=1)[0]

        if elems[0] not in [x[0] for x in holdout_pair] and not include_identity:
            break
        elif elems[0] in [x[0] for x in holdout_pair] and include_identity:
            break
    
    # Remove the holdout pair from available pairs
    available_pairs = all_pairs.copy()
    available_pairs.remove(holdout_pair)
        
    # Remove the reverse pair
    reverse_pair = (holdout_pair[1], holdout_pair[0])
    if reverse_pair in available_pairs:    
        available_pairs.remove(reverse_pair)

    held_out = [holdout_pair, reverse_pair]

    # Find facts that share elements with the holdout
    shares_left = [(a, b) for a, b in available_pairs if a == holdout_pair[0]]
    shares_right = [(a, b) for a, b in available_pairs if b == holdout_pair[1]]
    left_right_facts = shares_left + shares_right
    
    k_shots = len(left_right_facts) if k_shots is None else k_shots
    num_facts_needed = k_shots - 1
    
    sequence = []
    # Start with one copy of each fact (ensuring all facts appear at least once)
    for i in range(len(left_right_facts)):
        a, b = left_right_facts[i]
        c = (a[0] * b[0], a[1])
        sequence.append([',', wordfor[a], wordfor[b], '=', wordfor[c]])
    # Fill remaining slots with random samples from left_right_facts
    for i in range(num_facts_needed - len(left_right_facts)):
        a, b = task.prng.choice(left_right_facts)
        c = (a[0] * b[0], a[1])
        sequence.append([',', wordfor[a], wordfor[b], '=', wordfor[c]])
    random.shuffle(sequence)
    sequence = sum(sequence, [])
    # End with the held-out fact
    if held_out:
        a, b = held_out[0]
        c = (a[0] * b[0], a[1])
        sequence.extend([',', wordfor[a], wordfor[b], '=', wordfor[c]])
    
    return ''.join(sequence), [group], [group.order()], [vocab]

def construct_associative_sequence(task, group=None, k_shots=None, include_identity=False, unshuffled=False, num_triplets=1):
    """
    Constructs a sequence that requires associative reasoning to solve.
    
    Creates a sequence of facts from a group where the held-out query fact can be
    derived through associative property of group operations.
    
    Args:
        task: Task object with vocabulary and sampling methods.
        group: Group object to sample from. If None, samples a random group.
               Can also be a list containing a single group.
        k_shots (int, optional): Number of facts in the sequence. If None, uses all
                                 facts from the selected triplets.
        include_identity (bool): If True, allows identity element in holdout pair.
                                If False, excludes it.
        unshuffled (bool): If True, uses ordered vocabulary. If False, randomly
                          samples vocabulary.
        num_triplets (int): Number of associative triplets to include in the sequence.
    
    Returns:
        tuple: (sequence_str, [group], [order], [vocab])
            - sequence_str (str): Comma-separated string of facts
            - group (list): List containing the group used
            - order (list): List containing the group order
            - vocab (list): List containing the vocabulary string used
    """
    if group is None:
        group = task.sample_groups()[0]
    elif isinstance(group, list):
        group = group[0]

    # Create all possible pairs from single group
    elems = [(g, 0) for g in group.generate()]
    all_pairs = [(a, b) for a in elems for b in elems]

    # Create vocabulary mapping
    if unshuffled:
        vocab = ''.join(task.vocab[:group.order()])
    else:
        vocab = ''.join(random.sample(task.vocab[:16], k=group.order()))
    wordfor = {g: vocab[i] for i, g in enumerate(elems)}
    elemfor = {vocab[i]: g for i, g in enumerate(elems)}

    # Sample a holdout fact that has valid associative triplets
    while True:
        holdout_pair = random.sample(all_pairs, k=1)[0]

        holdout_ints = vocab.index(wordfor[holdout_pair[0]]), vocab.index(wordfor[holdout_pair[1]])
        triplets = determine_associative_pairs(holdout_ints, group, drop_duplicates=False)
        
        # Filter out triplets that contain the holdout pair or its reverse
        check_copy = lambda x, y: x == y or x == y[::-1]
        filtered_triplets = [x for x in triplets if not check_copy(x[0], holdout_ints) and 
                            not check_copy(x[1], holdout_ints) and 
                            not check_copy(x[2], holdout_ints)]

        if elems[0] not in holdout_pair and not include_identity and len(filtered_triplets) > 0:
            break
        elif elems[0] in holdout_pair and include_identity and len(filtered_triplets) > 0:
            break
    
    # Adjust num_triplets if we don't have enough
    if num_triplets > len(filtered_triplets):
        num_triplets = len(filtered_triplets)
        # print(f"Warning, chose too many triplets, there are only {len(filtered_triplets)}: \n{filtered_triplets}")
    filtered_triplets = random.sample(filtered_triplets, k=num_triplets)
    
    if k_shots is None:
        k_shots = len(sum(filtered_triplets, ())) + 1

    held_out = [holdout_pair]

    random.shuffle(filtered_triplets)

    # Convert triplets back to element pairs
    facts = [(elemfor[vocab[x[0]]], elemfor[vocab[x[1]]]) for x in sum(filtered_triplets, ())]
    num_facts_needed = k_shots - (1 if held_out else 0)

    sequence = []
    # Start with one copy of each fact (ensuring all facts appear at least once)
    for i in range(len(facts)):
        a, b = facts[i]
        c = (a[0] * b[0], a[1])
        sequence.append([',', wordfor[a], wordfor[b], '=', wordfor[c]])
    # Fill remaining slots with random samples from facts
    for i in range(num_facts_needed - len(facts)):
        a, b = task.prng.choice(facts)
        c = (a[0] * b[0], a[1])
        sequence.append([',', wordfor[a], wordfor[b], '=', wordfor[c]])
    random.shuffle(sequence)
    sequence = sum(sequence, [])
    # End with the held-out fact
    if held_out:
        a, b = held_out[0]
        c = (a[0] * b[0], a[1])
        sequence.extend([',', wordfor[a], wordfor[b], '=', wordfor[c]])    
    return ''.join(sequence), [group], [group.order()], [vocab]

def sample_distribution_sequence(task, k_shots: int = 200, distribution: str = None, unshuffled: bool = False, fixed_groups=None, **distribution_kwargs):
    """
    Samples a sequence from a specified distribution type.
    
    Args:
        task: Task object with vocabulary and sampling methods.
        k_shots (int): Number of facts in the sequence.
        distribution (str): Type of distribution. Options:
            - 'copy': Query can be solved by direct copying
            - 'commute': Query can be solved by commutative property
            - 'identity': Query can be solved by identity recognition
            - 'cancel': Query requires cancellation reasoning
            - 'associate': Query requires associative reasoning
            - 'test': Query cannot be solved by copy, commute
            - 'other': Query cannot be solved by copy, commute, or identity
            - None: Generic training data (no constraints)
        unshuffled (bool): If True, uses ordered vocabulary. If False, randomly
                          samples vocabulary.
        fixed_groups: Group(s) to use instead of sampling randomly.
        **distribution_kwargs: Additional arguments passed to specific constructors
                              (e.g., num_triplets for 'associate').
    
    Returns:
        tuple: (sequence_str, [group], [order], [vocab])
    
    Raises:
        NotImplementedError: If k_shots < 5 and suitable sequence cannot be found.
    """
    if distribution == "copy":
        exact_out = False
        commute_out = True
        condition = check_copyable

    elif distribution == "commute":
        exact_out = True
        commute_out = False
        condition = lambda s: not check_copyable(s) and check_reverse_copyable(s)

    elif distribution == "identity":
        exact_out = True
        commute_out = True
        condition = lambda s: not check_copyable(s) and not check_reverse_copyable(s) and check_identity(s)
    
    elif distribution == "cancel":
        return construct_cancellation_sequence(task, group=fixed_groups, k_shots=k_shots, unshuffled=unshuffled)
    
    elif distribution == 'associate':
        return construct_associative_sequence(task, group=fixed_groups, k_shots=k_shots, unshuffled=unshuffled, **distribution_kwargs)

    elif distribution == 'test':
        exact_out = True
        commute_out = True
        condition = lambda s: not check_copyable(s) and not check_reverse_copyable(s)

    elif distribution == 'other':
        exact_out = True
        commute_out = True
        condition = lambda s: not check_copyable(s) and not check_reverse_copyable(s) and not check_identity(s)

    else:  # Generic training data
        exact_out = False
        commute_out = False
        condition = lambda s: True

    counter = 0
    while True:
        counter += 1
        s, g, o, v = task.sample_run(k_shots, hold_out=exact_out, commute_out=commute_out, unshuffled=unshuffled, fixed_groups=fixed_groups)
        
        if condition(s):
            break
        elif k_shots < 5 and counter > 10:
            raise NotImplementedError("Need to evaluate on longer sequences (5+ facts)")
    
    return s, g, o, v

def sample_distribution_batch(task, batch_size: int, k_shots: int = 200, distribution=None, unshuffled: bool | str = False, fixed_groups: list = None, **distribution_kwargs):
    """
    Samples a batch of sequences from a specified distribution.
    
    Args:
        task: Task object with vocabulary and sampling methods.
        batch_size (int): Number of sequences to generate.
        k_shots (int): Number of facts in each sequence.
        distribution (str, optional): Type of distribution. See sample_distribution_sequence
                                     for available options.
        unshuffled (bool | str): If True, uses ordered vocabulary. If False, randomly
                                samples vocabulary.
        fixed_groups (list, optional): Group(s) to use instead of sampling randomly.
        **distribution_kwargs: Additional arguments passed to sequence constructors.
    
    Returns:
        dict: Dictionary containing:
            - inputs (torch.Tensor): Input sequences (batch_size, seq_len-1)
            - targets (torch.Tensor): Target sequences (batch_size, seq_len-1)
            - groups (tuple): Tuple of groups used for each sequence
            - orders (tuple): Tuple of group orders for each sequence
            - vocabulary (list): List of vocabulary strings for each sequence
            - prediction_mask (torch.Tensor): Boolean mask indicating prediction positions
    """
    if distribution not in ['copy', 'commute', 'identity', 'cancel', 'associate', 'test', 'other']:
        print(f"Warning! {distribution} is not a specified distribution. Sampling generic training data.")

    expressions, g, o, v = zip(*[sample_distribution_sequence(task, k_shots, distribution, unshuffled, fixed_groups, **distribution_kwargs)
                for _ in range(batch_size)])
    tensor = task.tensor_from_expression(expressions)

    return {
        "inputs": tensor[:,:-1],
        "targets": tensor[:,1:],
        "groups": g,
        "orders": o,
        "vocabulary": [''.join(voc) for voc in v],
        "prediction_mask": (tensor[:,:-1] == task.predict_token_id)
    }


### 
# Counterfactual Pair Sequence Construction
###

def sample_copy_counterfactual_pair(task, k_shots, fixed_groups=None, unshuffled=False, copy_distribution='copy', flipped_query=False):
    """
    
    """
    valid = False
    while not valid:

        copy_sequence, g_copy, o_copy, v_copy = sample_distribution_sequence(task, k_shots=k_shots, distribution=copy_distribution, unshuffled=unshuffled, fixed_groups=fixed_groups)
        
        if flipped_query:
            a,b = copy_sequence[-4:-2]
            elems = sum([[(g, i) for g in G.generate()] for i, G in enumerate(g_copy)], [])
            a_idx, b_idx = v_copy.index(a), v_copy.index(b)
            c_idx = elems.index((elems[b_idx][0] * elems[a_idx][0], elems[a_idx][1]))
            no_copy_sequence = copy_sequence[:-4] + b + a + '=' + v_copy[c_idx]
            v_no_copy = v_copy

            # Make sure query is not a square
            not_square = a != b           
            valid = not_square

        else:
            no_copy_sequence, g_no_copy, o_no_copy, v_no_copy = sample_distribution_sequence(task, k_shots=k_shots, distribution='test', unshuffled=unshuffled, fixed_groups=fixed_groups) # no copying possible

            # Make sure answers are different and queries are different
            not_same_answer = copy_sequence[-1] != no_copy_sequence[-1]
            not_same_query = copy_sequence[-4:-2] != no_copy_sequence[-4:-2]           

            valid = not_same_answer and not_same_query

    return copy_sequence, no_copy_sequence, v_copy, v_no_copy
