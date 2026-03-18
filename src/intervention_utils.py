import torch


def logit_lens(state, model, softmax=True):
    """
    Apply the logit lens technique to inspect intermediate model states.
    
    The logit lens projects a hidden state through the model's final layer norm
    and language model head to see what tokens the model would predict at that
    intermediate layer.
    
    Args:
        state (torch.Tensor): Hidden state tensor to project to vocabulary space.
            Shape: (batch_size, seq_len, hidden_dim)
        model: The transformer model containing ln_f and lm_head components.
        softmax (bool, optional): Whether to apply softmax to get probabilities.
            If False, returns raw logits. Defaults to True.
    
    Returns:
        torch.Tensor: Vocabulary predictions. If softmax=True, returns probability
            distribution over vocabulary. If softmax=False, returns raw logits.
            Shape: (batch_size, seq_len, vocab_size)
    """
    if softmax:
        return model.lm_head(model.transformer.ln_f(state)).softmax(-1)
    else:
        return model.lm_head(model.transformer.ln_f(state))
    
def attn_head_intervention_sweep(model, task, source, base, use_probs=False, token_idx=-1):
    """
    Perform activation patching across all attention heads to measure their causal effects.
    
    This function systematically patches each attention head's output from a source prompt
    into a base prompt, measuring how this intervention changes the model's predictions.
    This helps identify which attention heads are causally important for a specific behavior.
    
    Args:
        model: The transformer model to analyze.
        task: Task object with tensor_from_expression method to convert prompts to tensors.
        source (str): Source prompt whose attention head activations will be patched in.
        base (str): Base prompt that will receive the patched activations.
        use_probs (bool, optional): If True, compute differences in probabilities.
            If False, compute differences in logits. Defaults to False.
        token_idx (int, optional): Token position to patch. Defaults to -1 (last token).
    
    Returns:
        tuple: Contains:
            - scores (torch.Tensor): Intervention effects for each head.
                Shape: (n_layers, n_heads, vocab_size)
            - inputs_1 (torch.Tensor): Source prompt input tokens.
            - targets_1 (torch.Tensor): Source prompt target tokens.
            - inputs_2 (torch.Tensor): Base prompt input tokens.
            - targets_2 (torch.Tensor): Base prompt target tokens.
    """
    tensor_1 = task.tensor_from_expression([source])
    tensor_2 = task.tensor_from_expression([base])

    inputs_1, targets_1 = tensor_1[:, :-1], tensor_1[:, 1:]
    inputs_2, targets_2 = tensor_2[:, :-1], tensor_2[:, 1:]

    batch_size = tensor_1.shape[0]  
    scores = torch.zeros(model.config.n_layer, model.config.n_head, model.config.vocab_size)

    with torch.no_grad():
        for layer_index in range(model.config.n_layer):
            for head_index in range(model.config.n_head):

                with model.trace(inputs_1):
                    attn_in = model.transformer.h[layer_index].attn.c_proj.input
                    x_1 = attn_in.reshape(batch_size, -1, model.config.n_head, model.config.n_embd // model.config.n_head).save()
                    
                with model.trace(inputs_2):
                    clean_out = model.lm_head.output[:,-1].save()

                with model.trace(inputs_2) as tracer:
                    attn_in = model.transformer.h[layer_index].attn.c_proj.input
                    x_2 = attn_in.reshape(batch_size, -1, model.config.n_head, model.config.n_embd // model.config.n_head)
                    x_2[:,token_idx,head_index, :] = x_1[:,token_idx,head_index, :]
                    interv_out = model.lm_head.output[:,-1].save()
                
                if use_probs:
                    score = (interv_out.softmax(-1) - clean_out.softmax(-1))
                else:
                    score = (interv_out - clean_out)
                scores[layer_index, head_index] = score
    
    return scores, inputs_1, targets_1, inputs_2, targets_2


def head_output_to_vocab(model, inputs, head, token_idx=-1):
    """
    Project a single attention head's output directly to vocabulary space.
    
    This function isolates a specific attention head's output and projects it through
    the output projection and language model head to see what that head alone would
    predict. Useful for understanding individual head behaviors.
    
    Args:
        model: The transformer model to analyze.
        inputs (torch.Tensor): Input token tensor. Shape: (batch_size, seq_len)
        head (tuple): Tuple of (layer_index, head_index) specifying which attention
            head to analyze.
        token_idx (int, optional): Token position to extract head output from.
            Defaults to -1 (last token).
    
    Returns:
        torch.Tensor: Vocabulary predictions from this head's output.
            Shape: (batch_size, vocab_size)
    """
    if isinstance(head, tuple):
        layer_index, head_index = head

    logit_lens = lambda x: model.lm_head(model.transformer.ln_f(x))

    batch_size = inputs.shape[0]
    head_dim = model.config.n_embd // model.config.n_head
    head_out = torch.zeros((batch_size, model.config.n_embd)).to(model.device)

    with model.trace(inputs):
        attn_in = model.transformer.h[layer_index].attn.c_proj.input
        x_1 = attn_in.reshape(batch_size, -1, model.config.n_head, head_dim)
        head_out[:, head_index*head_dim:(head_index+1)*head_dim] = x_1[:, token_idx, head_index].save()

    head_out_proj = model.transformer.h[layer_index].attn.c_proj(head_out)

    return logit_lens(head_out_proj)


def cache_activations(model, inputs, model_component, model_component_str, cache_output=True):
    """
    Cache activations from a specific model component for later use.
    
    This function runs a forward pass and saves activations from a specified
    component, storing them on CPU to free GPU memory. Useful for activation
    patching experiments where you need to reuse activations multiple times.
    
    Args:
        model: The transformer model to run.
        inputs (torch.Tensor): Input token tensor. Shape: (batch_size, seq_len)
        model_component: The model component to cache (e.g., model.transformer.h[0]).
        model_component_str (str): String identifier for this component in the cache dict.
        cache_output (bool, optional): If True, cache the component's output.
            If False, cache the component's input. Defaults to True.
    
    Returns:
        dict: Dictionary mapping component name to cached activations (on CPU).
            Keys are the model_component_str, values are torch.Tensors.
    """
    cuda_cache = {}

    with torch.no_grad(), model.trace(inputs):
        if cache_output:
            cuda_cache[model_component_str] = model_component.output.save()
        else:
            cuda_cache[model_component_str] = model_component.input.save()

    cache = {k: v.cpu() for k, v in cuda_cache.items()}
    del cuda_cache
    torch.cuda.empty_cache()

    return cache


def inference_w_subspace_patch(model, model_component, prompts, W_proj, cache, intervention_index=-1):
    """
    Perform inference with subspace-based activation patching.
    
    This function patches activations within along specific directions defined by a projection
    matrix W_proj. It replaces components of the activation in the subspace defined by
    W_proj with cached counterfactual activations, while keeping components orthogonal
    to that subspace unchanged.
    
    Args:
        model: The transformer model to run.
        model_component: The model component to intervene on (e.g., model.transformer.h[0]).
        prompts (torch.Tensor): Input token tensor. Shape: (batch_size, seq_len)
        W_proj (torch.Tensor): Projection matrix defining the subspace to patch.
            Shape: (hidden_dim, hidden_dim). Should be a projection matrix (W_proj @ W_proj = W_proj).
        cache (torch.Tensor): Cached counterfactual activations to patch in.
            Shape: (batch_size, seq_len, hidden_dim)
        intervention_index (int, optional): Token position to apply the patch.
            Defaults to -1 (last token).
    
    Returns:
        torch.Tensor: Model logits after the intervention.
            Shape: (batch_size, seq_len, vocab_size)
    """
    I = torch.eye(W_proj.shape[0]).to(model.device)    
        
    with model.trace(prompts) as tracer:
        # Subspace patching along masked directions
        h_original = model_component.output[:,intervention_index].clone()
        h_counterfactual = cache[:,intervention_index].to(model.device)
        h_replace = torch.einsum('ij, nj -> ni', W_proj, h_counterfactual)
        h_keep = torch.einsum('ij, nj -> ni', (I - W_proj), h_original)
        model_component.output[:,intervention_index] = h_replace + h_keep
        logits = model.lm_head.output.save()
    
    return logits