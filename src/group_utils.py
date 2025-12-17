import torch
import random
import json
from types import SimpleNamespace
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics import CyclicGroup, DihedralGroup


def compose(pair, elements):
    """
    Computes the product of two group elements and returns its index, assuming no duplicates
    
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

def label_facts(sequence, token_idx, vocabulary):
    """
    Labels structural and identity-related properties of a token within a sequence. 
    
    Parameters:
        sequence (list): The input sequence of tokens.
        token_idx (int): Index of the token to analyze.
        vocabulary (list): List of possible tokens.
        
    Returns:
        SimpleNamespace: A structured representation of token-level and fact-level attributes, 
                         including slot type, identity properties, commutativity, and Cayley table index.
    """

    ### TOKEN LEVEL INFORMATION
    # slot
    if sequence[token_idx] in ['=', ',']:
        slot = sequence[token_idx]

        fact_start = token_idx-4
        if sequence[token_idx] == '=':
            fact_start = token_idx-2
      
    elif sequence[token_idx-1] == ',':
        slot = 'a'
        fact_start = token_idx
    elif sequence[token_idx-1] == '=':
        slot = 'c'
        fact_start = token_idx-3
    else:
        slot = 'b'
        fact_start = token_idx-1
    
    # Note: Currently, index 0 (the first ',') is not really attached to a fact
    if fact_start < 0 and fact_start+4 >=0:
        fact = sequence[fact_start:]
    else:
        fact = sequence[fact_start:fact_start+4]

    # symbol 
    symbol = sequence[token_idx]

    # cayley table index
    if symbol in ['=', ',']:
        cayley_idx = None
    else:
        cayley_idx = vocabulary.index(sequence[token_idx])


    # slots
    a_slot, b_slot, _, c_slot = list(fact)

    # identity_type
    identity_type = 'none'
    if a_slot == c_slot:
        identity_type = 'right'
    if b_slot == c_slot:
        identity_type = 'left'
    if identity_type != 'none' and a_slot == b_slot:
        identity_type = 'perfect'

    # is_identity 
    is_identity = False
    if identity_type != 'none':
        if identity_type == 'left' and slot == 'a':
            is_identity = True
        if identity_type == 'right' and slot == 'b':
            is_identity = True
        if identity_type == 'perfect' and slot == 'c':
            is_identity = True


    ### FACT LEVEL INFORMATION

    fact = fact
    identity = True if identity_type != 'none' else False
    identity_type = identity_type

    seen_ab = False 
    if fact in sequence[:fact_start]:
        seen_ab = True

    seen_ba = False
    commutative = False
    if f"{fact[1]}{fact[0]}{fact[2]}" in sequence[:fact_start]:
        seen_ba = True
        if f"{fact[1]}{fact[0]}{fact[2]}{fact[3]}" in sequence[:fact_start]:
            commutative = True

    square = False
    if a_slot == b_slot:
        square = True

    square_type = 'none'
    if square:
        square_type = 'regular'
        if a_slot == c_slot:
            square_type = 'idempotent'

    labels = {
        'token':{
            'symbol': symbol,
            'slot': slot,
            'cayley_index': cayley_idx,
            'a_slot': a_slot,
            'b_slot': b_slot,
            'c_slot': c_slot,
            'is_identity': is_identity,
        },
        'fact':{
            'symbol': fact,
            'is_identity': identity,
            'identity_type': identity_type,
            'seen_ab': seen_ab,
            'seen_ba': seen_ba,
            'is_commutative': commutative,
            'is_square': square,
            'square_type': square_type
        }
    }
    labels = json.loads(json.dumps(labels), object_hook=lambda d: SimpleNamespace(**d))
    return labels

def summarize_attention_pattern(sequence, token_idx, attention_scores, vocabulary, correct_answer=None):
    """
    Analyzes attention pattern for a token by computing summary statistics about what it attends to.
    
    Parameters:
        sequence (list): The input sequence of tokens
        token_idx (int): Index of the current token
        attention_scores (torch.Tensor): Attention weights over previous tokens
        vocabulary (list): List of possible tokens
        correct_answer (str, optional): The correct answer for the current fact
        
    Returns:
        SimpleNamespace: Summary statistics about the attention pattern
    """
    # Get labels for current token
    current = label_facts(sequence, token_idx, vocabulary)
    
    # Initialize statistics
    stats = {
        'slot_attention': {slot: 0.0 for slot in ['a', 'b', 'c', '=', ',']},
        'where_attention':{
            'current_fact': 0.0,
            'previous_facts': 0.0
            },
        'symbol_attention': {
            **{symbol: 0.0 for symbol in vocabulary},  # Add each vocabulary symbol
            '=': 0.0,
            ',': 0.0
        },
        'previous_fact_attention': {
            'shares_a_left': 0.0,    # ax=?
            'shares_b_right': 0.0,   # xb=?
            'a_on_right': 0.0,       # xa=?
            'b_on_left': 0.0,        # bx=?   
            'result_is_a': 0.0,      # xy=a
            'result_is_b': 0.0,      # xy=b
            'reverse_pair': 0.0,     # ba=? for current ab=?
            'squares': 0.0,          # xx=?
            'idempotent': 0.0,       # xx=x
            'correct_answer': 0.0,
        }
    }
    
    total_attention = attention_scores.sum().item()
    if total_attention == 0:
        return SimpleNamespace(**stats)
    
    # Analyze attention to each previous token
    for prev_idx in range(len(attention_scores)):
        score = attention_scores[prev_idx].item()
        if score == 0:
            continue
            
        prev = label_facts(sequence, prev_idx, vocabulary)
        
        # Slot attention
        stats['slot_attention'][prev.token.slot] += score
        
        # Symbol attention - track every symbol
        stats['symbol_attention'][prev.token.symbol] += score
        
        # Where attention
        if prev.fact.symbol == current.fact.symbol:
            stats['where_attention']['current_fact'] += score
        else:
            stats['where_attention']['previous_facts'] += score
        
        # Related fact patterns
        if prev_idx < len(attention_scores) - 1:
        
            # Correct answer attention
            if correct_answer is not None and prev.token.symbol == correct_answer:
                stats['previous_fact_attention']['correct_answer'] += score

            # For a?=?
            if prev.token.a_slot == current.token.a_slot:
                stats['previous_fact_attention']['shares_a_left'] += score
                
            # For ?b=?
            if prev.token.b_slot == current.token.b_slot:
                stats['previous_fact_attention']['shares_b_right'] += score
            
            # For ?a=?
            if prev.token.b_slot == current.token.a_slot:
                stats['previous_fact_attention']['a_on_right'] += score

            # For b?=?
            if prev.token.a_slot == current.token.b_slot:
                stats['previous_fact_attention']['b_on_left'] += score

            # For ??=a
            if prev.token.c_slot == current.token.a_slot:
                stats['previous_fact_attention']['result_is_a'] += score
                
            # For ??=b
            if prev.token.c_slot == current.token.b_slot:
                stats['previous_fact_attention']['result_is_b'] += score
                
            # Reverse pairs
            if (prev.token.a_slot == current.token.b_slot and 
                prev.token.b_slot == current.token.a_slot):
                stats['previous_fact_attention']['reverse_pair'] += score
                
            # Squares and idempotent
            if prev.fact.is_square:
                stats['previous_fact_attention']['squares'] += score
                if prev.fact.square_type == 'idempotent':
                    stats['previous_fact_attention']['idempotent'] += score
    
    # Normalize all scores by total attention
    for category in stats.values():
        for key in category:
            category[key] /= total_attention
            
    return SimpleNamespace(**stats)