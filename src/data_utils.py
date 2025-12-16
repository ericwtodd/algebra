import random
import torch

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


def compose(pair, elements):
    """
    Computes the product of two group elements and returns its index.
    
    Args:
        pair (tuple): A pair of indices (a, b) representing positions in elements list.
        elements (list): List of group elements ordered by their indexing.
    
    Returns:
        int: Index of the product elements[a] * elements[b] in the elements list.
    """
    a, b = pair
    return elements.index(elements[a] * elements[b])


def generate_cayley_table(group):
    """
    Generates the Cayley table of a finite group.
    
    Args:
        group (sympy.combinatorics.perm_groups.PermutationGroup): The group for
            which to generate the Cayley table.
    
    Returns:
        torch.LongTensor: Tensor of shape (order, order) where entry (i, j) is the
                         index of the product of the i-th and j-th elements.
    """
    order = group.order()
    elements = list(group.generate())
    
    cayley_table = []
    for i in range(order):
        row = []
        for j in range(order):
            row.append(compose((i, j), elements))
        cayley_table.append(row)
    
    return torch.LongTensor(cayley_table)


def determine_associative_pairs(pair, group, drop_X=False, drop_R=False, drop_duplicates=True):
    """
    Finds all triples of element-pairs that satisfy associativity for a given pair.
    
    Given a pair (a, b) in the group, finds all possible triples ((x, y), (y, z), (w, z))
    satisfying one of two associativity forms:
    1. (a * g = f, g * d = b, f * d = c) leading to a * b = c
    2. (d * b = f, g * d = a, g * f = c) leading to a * b = c
    
    Args:
        pair (tuple): A pair (a, b) of element indices in the group.
        group (sympy.combinatorics.perm_groups.PermutationGroup): The group containing
            the elements.
        drop_X (bool): If True, removes triples containing the original pair (a, b).
        drop_R (bool): If True, removes triples containing the reverse pair (b, a).
        drop_duplicates (bool): If True, removes duplicate pairs within each triplet.
    
    Returns:
        list: List of triples, where each triple is a tuple of three element-pairs
              satisfying associativity.
    """
    elements = list(group.generate())
    cayley_table = generate_cayley_table(group)
        
    a, b = pair
    c = compose(pair, elements)

    # Determine the (x, y) candidates for xy = c
    xy_c_candidates = [(i.item(), j.item()) for i, j in zip(*torch.where(cayley_table == c))]

    triples = []
    # Associativity v1: (a*g=f, g*d=b, f*d=c) -> ab=c
    for (f, d) in xy_c_candidates:
        g = torch.where(cayley_table[:, d] == b)[0].item()
        g2 = torch.where(cayley_table[a, :] == f)[0].item()
        assert g == g2
        assert compose((a, g), elements) == f
        assert compose((g, d), elements) == b
        assert compose((f, d), elements) == c
        if drop_duplicates:
            triples.append(tuple(set(((a, g), (g, d), (f, d)))))
        else:
            triples.append(tuple(((a, g), (g, d), (f, d))))
        
    # Associativity v2: (d*b=f, g*d=a, g*f=c) -> ab=c
    for (g, f) in xy_c_candidates:
        d = torch.where(cayley_table[g, :] == a)[0].item()
        d2 = torch.where(cayley_table[:, b] == f)[0].item()
        assert d == d2
        assert compose((d, b), elements) == f
        assert compose((g, d), elements) == a
        assert compose((g, f), elements) == c
        if drop_duplicates:
            triples.append(tuple(set(((d, b), (g, d), (g, f)))))
        else:
            triples.append(tuple(((d, b), (g, d), (g, f))))
    
    if drop_X:
        triples = [x for x in triples if (a, b) not in x]
    if drop_R:
        triples = [x for x in triples if (b, a) not in x]
    
    return triples


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