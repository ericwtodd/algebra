import torch


def logit_lens(state, model, softmax=True):
    """
    """
    if softmax:
        return model.lm_head(model.transformer.ln_f(state)).softmax(-1)
    else:
        return model.lm_head(model.transformer.ln_f(state))
    
def attn_head_intervention_sweep(model, task, source, base, use_probs=False, token_idx=-1):
    """
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
                    score = (interv_out.softmax(-1) - clean_out.softmax(-1))#.index_select(-1, torch.LongTensor(track_token).to('cuda')).item()
                else: #logits
                    score = (interv_out - clean_out)#.index_select(-1, torch.LongTensor(track_token).to('cuda')).item()
                scores[layer_index, head_index] = score
    
    return scores, inputs_1, targets_1, inputs_2, targets_2


def head_output_to_vocab(model, inputs, head, token_idx=-1):
    """
    """
    if isinstance(head, tuple):
        layer_index, head_index = head

    logit_lens = lambda x: model.lm_head(model.transformer.ln_f(x))

    hidden_size = model.config.n_embd
    head_dim = hidden_size // model.config.n_head
    head_out = torch.zeros((1,model.config.n_embd)).to(model.device)

    with model.trace(inputs):
        attn_in = model.transformer.h[layer_index].attn.c_proj.input
        x_1 = attn_in.reshape(1, -1, model.config.n_head, model.config.n_embd // model.config.n_head)
        head_out[:,head_index*head_dim:(head_index+1)*head_dim] = x_1[:,token_idx,head_index].save()
    
    head_out_proj = model.transformer.h[layer_index].attn.c_proj(head_out)

    return logit_lens(head_out_proj)