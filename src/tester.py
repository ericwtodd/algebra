import torch
import torch.nn.functional as F
import random
from sympy.combinatorics import CyclicGroup, DihedralGroup
from collections import defaultdict
from .data_utils import sample_distribution_batch

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


class Tester:
    """
    Handles model evaluation across multiple diagnostic metrics.
    
    Evaluates models on various algebraic reasoning tasks including:
    - Standard accuracy on held-out test sets
    - Copy/commutative accuracy
    - Identity recognition
    - Cancellation reasoning
    - Associativity
    - Closure properties
    - Row/column elimination
    """
    
    def __init__(self, config):
        """
        Initialize the Tester.
        
        Args:
            config: Configuration object containing evaluation parameters like
                   batch_size, k_shots, evaluation_size, etc.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_batches = []
        self.config = config
    
    def get_test_batches(self, task, context_length):
        """
        Generate and cache test batches for evaluation.
        
        Args:
            task: Task object with sample_batch method
            context_length (int): Maximum sequence length
        
        Returns:
            list: List of test batches
        """
        while len(self.test_batches) < self.config.evaluation_size / self.config.batch_size:
            self.test_batches.append(task.sample_batch(
                batch_size=self.config.batch_size,
                k_shots=self.config.k_shots,
                max_length=context_length,
                hold_out=True,
                unshuffled=True
            ))
        return self.test_batches
    
    def evaluate(self, model, task):
        """
        Run all evaluation metrics and return aggregated results.
        
        Args:
            model: PyTorch model to evaluate
            task: Task object providing data generation and evaluation utilities
        
        Returns:
            tuple: (main_accuracy, summary, results_dict)
                - main_accuracy (float): Overall accuracy on test set
                - summary (str): Human-readable summary of results
                - results_dict (dict): Dictionary of all metric values
        """
        model.eval()
        
        results = {}
        
        # Main accuracy on standard test set
        main_acc, summary = self._evaluate_main_accuracy(model, task)
        results['eval_accuracy'] = main_acc
        
        # Per-group breakdown
        # results.update(self._per_group_accuracies(model, task))
        
        # Diagnostic metrics
        results.update(self._copy_accuracy(model, task))
        results.update(self._identity_accuracy(model, task))
        results.update(self._cancellation_accuracy(model, task))
        results.update(self._closure_metrics(model, task))
        results.update(self._associativity_accuracy(model, task))
        # results.update(self._row_column_elimination_accuracy(model, task))
        
        return main_acc, summary, results
    
    # =========================================================================
    # MAIN ACCURACY
    # =========================================================================
    
    def _evaluate_main_accuracy(self, model, task):
        """
        Standard accuracy on held-out test set.
        
        Evaluates the model on randomly generated test sequences, measuring
        accuracy only on the final token prediction.
        
        Args:
            model: PyTorch model to evaluate
            task: Task object for data generation
        
        Returns:
            tuple: (accuracy, summary)
                - accuracy (float): Fraction of correct predictions
                - summary (str): Human-readable summary from task.summarize()
        """
        correct_t, total = 0, 0
        predictions = []
        
        with torch.no_grad():
            # Generate test batches
            test_batches = self.get_test_batches(task, model.config.block_size)
                
            for batch in test_batches:
                batch = {k: v.to(self.device) for k, v in batch.items() 
                        if isinstance(v, torch.Tensor)}
                
                # Apply padding if needed
                if self.config.leftpad:
                    inputs, targets = self._apply_padding(
                        batch["inputs"], batch["targets"], 
                        model.config.block_size, task.pad_token_id
                    )
                else:
                    inputs, targets = batch["inputs"], batch["targets"]
                
                # Forward pass
                outputs = model(inputs)
                if not isinstance(outputs, torch.Tensor):
                    outputs = outputs[0]
                
                # Accuracy on final token only
                masked_outputs = outputs[:, -1:]
                masked_targets = targets[:, -1:]

                correct, count = accuracy_fn(masked_outputs, masked_targets)
                correct_t += correct
                total += count
                
                # Accumulate predictions for use in the summary
                predictions.append(outputs.argmax(-1).cpu())
        
        accuracy = correct_t / total
        summary = task.summarize(test_batches, predictions, accuracy)
        return accuracy, summary
    
    def _apply_padding(self, inputs, targets, context_length, pad_token_id):
        """
        Apply left padding to inputs and targets to reach context_length.
        
        Args:
            inputs (torch.Tensor): Input sequences [batch_size, seq_len]
            targets (torch.Tensor): Target sequences [batch_size, seq_len]
            context_length (int): Desired sequence length
            pad_token_id (int): Token ID to use for padding
        
        Returns:
            tuple: (padded_inputs, padded_targets)
        """
        input_padding = context_length - inputs.size(1)
        target_padding = context_length - targets.size(1)
        padded_inputs = F.pad(inputs, (input_padding, 0), value=pad_token_id)
        padded_targets = F.pad(targets, (target_padding, 0), value=pad_token_id)
        return padded_inputs, padded_targets
       
    def _evaluate_final_token(self, model, batch):
        """
        Helper to evaluate accuracy at the final token only.
        
        Args:
            model: PyTorch model to evaluate
            batch (dict): Batch dictionary with 'inputs' and 'targets'
        
        Returns:
            float: Accuracy on final token predictions
        """
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        
        # Forward pass
        outputs = model(inputs)
        if not isinstance(outputs, torch.Tensor):
            outputs = outputs[0]
        
        masked_outputs = outputs[:, -1:]
        masked_targets = targets[:, -1:]
        
        correct, total = accuracy_fn(masked_outputs, masked_targets)
        return correct / total
    
    # =========================================================================
    # DIAGNOSTIC METRICS
    # =========================================================================

    def _copy_accuracy(self, model, task):
        """
        Accuracy on facts solvable by copying from context.
        
        Tests two types of copying:
        - Exact copying: Query (a,b) has appeared exactly as (a,b) before
        - Commutative copying: Query (a,b) has appeared as (b,a) before
        
        Args:
            model: PyTorch model to evaluate
            task: Task object for data generation
        
        Returns:
            dict: Dictionary with keys:
                - 'Performances/Copy_Exact_Accuracy'
                - 'Performances/Copy_Reverse_Accuracy'
        """
        exact_batch = sample_distribution_batch(
            task, 
            batch_size=self.config.batch_size//2,
            k_shots=self.config.k_shots,
            distribution='copy',
            unshuffled=False
        )
        
        rev_batch = sample_distribution_batch(
            task,
            batch_size=self.config.batch_size//2,
            k_shots=self.config.k_shots,
            distribution='commute',
            unshuffled=False
        )

        with torch.no_grad():
            exact_acc = self._evaluate_final_token(model, exact_batch)
            reverse_acc = self._evaluate_final_token(model, rev_batch)
        
        return {
            "Performances/Copy_Exact_Accuracy": exact_acc,
            "Performances/Copy_Reverse_Accuracy": reverse_acc,
        }
    
    def _identity_accuracy(self, model, task):
        """
        Accuracy on sequences where identity recognition solves the query.
        
        Tests if the model can recognize when a fact like xy=x indicates
        that y is an identity element.
        
        Args:
            model: PyTorch model to evaluate
            task: Task object for data generation
        
        Returns:
            dict: Dictionary with key 'Performances/Identity_Accuracy'
        """
        batch = sample_distribution_batch(
            task,
            batch_size=self.config.batch_size//2,
            k_shots=self.config.k_shots,
            distribution='identity',
            unshuffled=False
        )
        
        with torch.no_grad():
            accuracy = self._evaluate_final_token(model, batch)
        
        return {"Performances/Identity_Accuracy": accuracy}
    
    def _cancellation_accuracy(self, model, task):
        """
        Accuracy on sequences requiring cancellation reasoning.
        
        Tests if the model can derive answers by cancelling intermediate
        elements across multiple facts.
        
        Args:
            model: PyTorch model to evaluate
            task: Task object for data generation
        
        Returns:
            dict: Dictionary with key 'Performances/Cancellation_Accuracy'
        """
        batch = sample_distribution_batch(
            task,
            batch_size=self.config.batch_size//2,
            k_shots=50,  # Keeping this at 50 (largest group is order 10)
            distribution='cancel',
            unshuffled=False
        )
        
        with torch.no_grad():
            accuracy = self._evaluate_final_token(model, batch)
        
        return {"Performances/Cancellation_Accuracy": accuracy}
    
    def _associativity_accuracy(self, model, task):
        """
        Accuracy on sequences requiring associative reasoning.
        
        Tests if the model can compose facts associatively to derive new facts.
        Evaluates with both 1 and 2 associative triplets.
        
        Args:
            model: PyTorch model to evaluate
            task: Task object for data generation
        
        Returns:
            dict: Dictionary with keys:
                - 'Performances/Associativity_Accuracy_1_Triplet'
                - 'Performances/Associativity_Accuracy_2_Triplets'
        """
        batch = sample_distribution_batch(
            task,
            batch_size=self.config.batch_size//2,
            k_shots=50,  # Keeping this at 50 (largest group is order 10)
            distribution='associate',
            unshuffled=False,
            num_triplets=1
        )

        batch_2 = sample_distribution_batch(
            task,
            batch_size=self.config.batch_size//2,
            k_shots=50,  # Keeping this at 50 (largest group is order 10)
            distribution='associate',
            unshuffled=False,
            num_triplets=2
        )
        
        with torch.no_grad():
            accuracy_1 = self._evaluate_final_token(model, batch)
            accuracy_2 = self._evaluate_final_token(model, batch_2)
        
        return {
            "Performances/Associativity_Accuracy_1_Triplet": accuracy_1,
            "Performances/Associativity_Accuracy_2_Triplets": accuracy_2
        }

    def _closure_metrics(self, model, task):
        """
        Evaluate closure properties and structural token predictions.
        
        For sequences with facts like ",ab=c", evaluates:
        1. Closure: After seeing 'a', does the model predict valid 'b' values
           from the same group in its top-K predictions?
        2. Equals prediction: After ',ab' does it predict '='?
        3. Comma prediction: After ',ab=c' does it predict ','?
        
        Args:
            model: PyTorch model to evaluate
            task: Task object for data generation
        
        Returns:
            dict: Dictionary with keys:
                - 'Performances/Closure_Strict_TopK_Acc': Fraction where all top-K are valid
                - 'Performances/Closure_TopK_Pct': Average percentage of top-K that are valid
                - 'Performances/EqualsToken_Acc': Accuracy on '=' prediction
                - 'Performances/CommaToken_Acc': Accuracy on ',' prediction
        """
        SEP_ID = task.sep_token_id
        GAP_ID = task.predict_token_id
        
        with torch.no_grad():
            batch = task.sample_batch(
                batch_size=self.config.batch_size,
                k_shots=self.config.k_shots,
                max_length=model.config.block_size,
                hold_out=True,
                unshuffled=False,
            )
            
            inputs = batch["inputs"].to(self.device)
            targets = batch["targets"].to(self.device)
            logits = model(inputs)
            if not isinstance(logits, torch.Tensor):
                logits = logits[0]
            preds = logits.argmax(-1)
            
            # Counters for each metric
            a_strict_correct = 0
            a_pct_sum = 0.0
            equals_correct = 0
            comma_correct = 0
            
            # For each sequence in the batch
            for row in range(inputs.size(0)):
                vocab_str = batch["vocabulary"][row]
                seq_groups = batch["groups"][row]
                
                # Map: token_id -> group_index
                tok2grp = {}
                vocab_pos = 0
                for gi, G in enumerate(seq_groups):
                    for _ in range(G.order()):
                        char = vocab_str[vocab_pos]
                        tok2grp[task.numfor[char]] = gi
                        vocab_pos += 1
                
                # Track which tokens we've seen per group
                seen = [set() for _ in seq_groups]
                for p in range(inputs.size(1) - 1):
                    tok = int(inputs[row, p].item())
                    if tok in tok2grp:
                        seen[tok2grp[tok]].add(tok)
                
                # Find separator positions
                sep_positions = (inputs[row] == SEP_ID).nonzero(as_tuple=True)[0]
                if len(sep_positions) < 2:
                    continue
                
                final_sep = sep_positions[-1].item()
                second_to_last_sep = sep_positions[-2].item()
                
                # Need room for final fact: , a b =
                if final_sep + 3 >= inputs.size(1):
                    continue
                
                # What's the 'a' in the final fact?
                a_tok = int(targets[row, final_sep].item())
                gi = tok2grp.get(a_tok)
                if gi is None or not seen[gi]:
                    continue
                
                # === A-SLOT: Position final_sep+1 (predicting 'b' after 'a') ===
                K = len(seen[gi])
                topK = logits[row, final_sep + 1].argsort(descending=True)[:K].tolist()
                
                num_correct = sum(1 for tok in topK if tok in seen[gi])
                if num_correct == K:
                    a_strict_correct += 1
                a_pct_sum += num_correct / K
                
                # === EQUALS-SLOT: Position final_sep+2 (predicting '=' after 'a b') ===
                pred_equals = int(preds[row, final_sep + 2].item())
                if pred_equals == GAP_ID:
                    equals_correct += 1
                
                # === COMMA-SLOT: Position second_to_last_sep+4 (predicting ',' after 'c') ===
                if second_to_last_sep + 4 < inputs.size(1):
                    pred_comma = int(preds[row, second_to_last_sep + 4].item())
                    if pred_comma == SEP_ID:
                        comma_correct += 1
            
            batch_size = inputs.size(0)
            return {
                "Performances/Closure_Strict_TopK_Acc": a_strict_correct / batch_size,
                "Performances/Closure_TopK_Pct": a_pct_sum / batch_size,
                "Performances/EqualsToken_Acc": equals_correct / batch_size,
                "Performances/CommaToken_Acc": comma_correct / batch_size,
            }

    def _row_column_elimination_accuracy(self, model, task, col=True):
        """
        Tests row and column elimination reasoning.
        
        Provides all but one entry in a row (or column) of a group's Cayley table
        and asks the model to predict the missing entry. This tests whether the
        model understands that rows and columns are permutations.
        
        For row elimination: given facts (a,g₁), (a,g₂), ..., (a,gₙ₋₁), predict (a,b)
        For column elimination: given facts (g₁,b), (g₂,b), ..., (gₙ₋₁,b), predict (a,b)
        
        Args:
            model: PyTorch model to evaluate
            task: Task object for data generation
            col (bool): If True, also test column elimination
        
        Returns:
            dict: Dictionary with keys:
                - 'Performances/RowElim_Accuracy'
                - 'Performances/ColElim_Accuracy' (if col=True)
        """
        results = defaultdict(lambda: {"correct": 0, "total": 0})
        
        with torch.no_grad():
            # Test Cyclic groups 3-10 and Dihedral groups 2-5
            groups_to_test = [(CyclicGroup, "Cyclic", x) for x in range(3, 11)] + \
                            [(DihedralGroup, "Dihedral", x) for x in [2, 3, 4, 5]]
            
            for GroupClass, group_name, order in groups_to_test:
                G = GroupClass(order)
                elems = list(G.elements)
                
                for _ in range(self.config.batch_size//2):  # 64 trials per group
                    # Shuffle vocabulary
                    vocab = list(task.vocab[:task.num_symbols])
                    random.shuffle(vocab)
                    vocab = ''.join(vocab[:G.order()])
                    
                    # Pick random a, b
                    a, b = random.sample(elems, 2)
                    
                    # Map elements to tokens
                    word_for = {e: vocab[i] for i, e in enumerate(elems)}
                    tok = lambda e: task.numfor[word_for[e]]
                    
                    # === ROW ELIMINATION ===
                    prompt_row = []
                    for g in elems:
                        if g != b:
                            prompt_row.extend([task.sep_token_id, tok(a), tok(g), 
                                            task.predict_token_id, tok(a * g)])
                    
                    prompt_row.extend([task.sep_token_id, tok(a), tok(b), task.predict_token_id])
                    
                    ids = torch.tensor([prompt_row], dtype=torch.long, device=self.device)
                    logits = model(ids)
                    if not isinstance(logits, torch.Tensor):
                        logits = logits[0]
                    
                    pred = logits[0, -1].argmax().item()
                    results["row"]["correct"] += (pred == tok(a * b))
                    results["row"]["total"] += 1
                    
                    # === COLUMN ELIMINATION ===
                    if col:
                        prompt_col = []
                        for g in elems:
                            if g != a:
                                prompt_col.extend([task.sep_token_id, tok(g), tok(b), 
                                                task.predict_token_id, tok(g * b)])
                        
                        prompt_col.extend([task.sep_token_id, tok(a), tok(b), task.predict_token_id])
                        
                        ids = torch.tensor([prompt_col], dtype=torch.long, device=self.device)
                        logits = model(ids)
                        if not isinstance(logits, torch.Tensor):
                            logits = logits[0]
                        
                        pred = logits[0, -1].argmax().item()
                        results["col"]["correct"] += (pred == tok(a * b))
                        results["col"]["total"] += 1
        
        # Format output
        output = {
            "Performances/RowElim_Accuracy": results["row"]["correct"] / results["row"]["total"]
        }
        if col:
            output["Performances/ColElim_Accuracy"] = results["col"]["correct"] / results["col"]["total"]
        
        return output

    def _per_group_accuracies(self, model, task):
        """
        Compute accuracy for each group type individually.
        
        Evaluates the model separately on each group in the task's group
        distribution, measuring only final token accuracy.
        
        Args:
            model: PyTorch model to evaluate
            task: Task object for data generation
        
        Returns:
            dict: Dictionary mapping group names to accuracies, e.g.:
                - 'Performances/PerGroup_Cyclic_5_Accuracy'
                - 'Performances/PerGroup_Dihedral_3_Accuracy'
        """
        results = {}
        
        # Get all groups based on task type
        all_groups = task._all_groups()
        
        with torch.no_grad():
            for G in all_groups:
                # Generate batch for this specific group
                batch = task.sample_batch(
                    batch_size=self.config.batch_size,
                    k_shots=self.config.k_shots,
                    max_length=model.config.block_size,
                    hold_out=True,
                    unshuffled=True,
                    fixed_groups=[G]
                )
                
                acc = self._evaluate_final_token(model, batch)
            
                if G.is_cyclic:
                    g_name = f"Cyclic_{G.order()}"
                elif G.is_dihedral:
                    g_name = f"Dihedral_{G.order() // 2}"
                else:
                    g_name = f"Group_{G.order()}"
                
                results[f"Performances/PerGroup_{g_name}_Accuracy"] = acc
        
        return results
