import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from IPython.display import HTML, display

def colorize_tokens(text: str, probabilities: list, visibility_threshold: float = 0.05) -> str:
    """Create HTML with color-coded characters using direct probability-to-color mapping.
    
    Args:
        text: String where each character is a token
        probabilities: List of probabilities (0-1) for each character
        visibility_threshold: Minimum probability to show coloring (default 0.05)
    
    Returns:
        HTML string with color-coded characters in a styled container
    """
    html_parts = []
    
    for char, prob in zip(text, probabilities):
        # Only apply coloring if above visibility threshold
        if prob >= visibility_threshold:
            # Direct linear mapping: probability directly controls color intensity
            # White (255,255,255) to Blue (0,0,255)
            # As probability increases, red and green decrease linearly
            red_green = int(255 * (1 - prob))  # 255 to 0 as prob goes 0 to 1
            blue = 255  # Always full blue
            
            color = f"rgb({red_green}, {red_green}, {blue})"
            style = "font-weight: bold;" if prob > 0.8 else ""
        else:
            # Below threshold - no coloring
            color = "transparent"
            style = ""
        
        # Preserve spaces and special characters
        display_char = char if char != ' ' else '&nbsp;'
        
        # Add the span with appropriate styling
        if color != "transparent":
            # Text color: white for dark backgrounds (high probability), black for light
            text_color = "white" if prob > 0.5 else "black"
            html_parts.append(
                f'<span style="background-color: {color}; color: {text_color}; '
                f'padding: 1px 2px; border-radius: 3px; margin: 0.5px; {style}; '
                f'outline: 2px solid transparent; transition: outline-color 0.2s;" '
                f'onmouseover="this.style.outlineColor=\'#333\'" '
                f'onmouseout="this.style.outlineColor=\'transparent\'" '
                f'title="Token: \'{char}\' | Probability: {prob:.3f}">{display_char}</span>'
            )
        else:
            html_parts.append(
                f'<span style="padding: 1px 2px; margin: 0.5px; '
                f'outline: 2px solid transparent; transition: outline-color 0.2s;" '
                f'onmouseover="this.style.outlineColor=\'#333\'" '
                f'onmouseout="this.style.outlineColor=\'transparent\'" '
                f'title="Token: \'{char}\' | Probability: {prob:.3f}">{display_char}</span>'
            )
    
    # Wrap everything in a styled container
    container_html = f'''
    <div style="
        background-color: #f6f8fa;
        border: 1px solid #d1d9e0;
        border-radius: 6px;
        padding: 4px 12px;
        margin: 10px 0;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 14px;
        line-height: 1.4;
        word-wrap: break-word;
        overflow-x: auto;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        max-width: 475px;
    ">
        {"".join(html_parts)}
    </div>
    '''
    
    return container_html



def collect_pca_activations(model, task, groups, n_samples_per_group, n_shots,
                             modules, token_idx=-1, distribution=None):
    """
    Collect activations from model modules for PCA analysis.

    Runs one forward pass per sample, saving activations from all specified modules
    in a single trace (efficient multi-layer collection).

    Args:
        model: NNsight-wrapped model.
        task: Task object with tensor_from_expression method.
        groups (list): List of sympy group objects to sample from.
        n_samples_per_group (int): Number of sequences to generate per group.
        n_shots (int): Sequence length (number of facts).
        modules: Single nnsight module envoy, OR dict {key: envoy}.
            Each module's .output[0, positions, :] is saved.
        token_idx (int or None):
            int  -- collect that single position per sample -> activations [N, D],
                    labels are simple dicts with 'group'.
            None -- collect all '=' positions (every 5th token starting at 3) per sample
                    -> activations [N*k, D], labels are label_facts objects with rich metadata.
        distribution (str or None): Passed to sample_distribution_sequence.
            Use 'identity', 'other', None (generic), etc.

    Returns:
        (activations, labels) where:
        - activations: tensor [N, D] if modules is a single envoy;
                       dict {key: tensor [N, D]} if modules is a dict.
        - labels: list of length N (token_idx is int) or N*k (token_idx is None).
    """
    from src.data_utils import sample_distribution_sequence
    from src.group_utils import label_facts

    single = not isinstance(modules, dict)
    module_dict = {0: modules} if single else modules

    # '=' positions: 3, 8, 13, ... within the input (which has n_shots*5 - 1 tokens)
    all_positions = list(range(3, n_shots * 5, 5))
    multi = (token_idx is None)

    all_acts = {k: [] for k in module_dict}
    labels = []

    with torch.no_grad():
        for group in groups:
            for _ in range(n_samples_per_group):
                seq, _, _, vocab = sample_distribution_sequence(task, n_shots, distribution,
                                                                fixed_groups=[group])
                inputs = task.tensor_from_expression([seq])[:, :-1]

                acts = {}
                if multi:
                    positions = [p for p in all_positions if p < inputs.shape[1]]
                    with model.trace(inputs):
                        for k in module_dict:
                            acts[k] = module_dict[k].output[0, positions].save()
                    for k in module_dict:
                        all_acts[k].append(acts[k].cpu())    # [len(positions), D]
                    vocab_str = vocab[0] if isinstance(vocab[0], str) else ''.join(vocab[0])
                    for p in positions:
                        labels.append(label_facts(list(seq), p, vocab_str))
                else:
                    with model.trace(inputs):
                        for k in module_dict:
                            acts[k] = module_dict[k].output[0, token_idx].save()
                    for k in module_dict:
                        all_acts[k].append(acts[k].cpu())    # [D]
                    labels.append({'group': group})

    if multi:
        result = {k: torch.cat(all_acts[k], dim=0) for k in module_dict}
    else:
        result = {k: torch.stack(all_acts[k]) for k in module_dict}

    if single:
        return result[0], labels
    return result, labels


def pca_fit(vecs_1, vecs_2=None, n_components=32):
    """
    Fit PCA jointly on one or two sets of activations.

    Args:
        vecs_1: Tensor [N, D], or dict {key: tensor [N, D]}.
        vecs_2: Tensor [N, D], or dict {key: tensor [N, D]}, or None.
            When provided, PCA is fit on the concatenation of vecs_1 and vecs_2.
        n_components (int): Number of PCA components to retain.

    Returns:
        Dict with same keys as vecs_1 (key 0 if plain tensors were passed).
        Each value is a dict with:
            - 'components': tensor [n_components, D]
            - 'explained_variance': tensor [n_components]
            - 'X_1': tensor [N, D]
            - 'X_2': tensor [N, D]  (only when vecs_2 is given)
    """
    single = not isinstance(vecs_1, dict)
    v1 = {0: vecs_1} if single else vecs_1
    v2 = ({0: vecs_2} if single else vecs_2) if vecs_2 is not None else None

    result = {}
    for k in v1:
        X1 = v1[k]
        entry = {'X_1': X1}

        if v2 is not None:
            X2 = v2[k]
            X = torch.cat([X1, X2], dim=0)
            entry['X_2'] = X2
        else:
            X = X1

        Xc = X - X.mean(0, keepdim=True)
        q = min(n_components, X.shape[1])
        _U, S, V = torch.pca_lowrank(Xc, q=q)
        entry['components'] = V[:, :q].T.contiguous()
        entry['explained_variance'] = (S[:q] ** 2) / (X.shape[0] - 1)
        result[k] = entry

    return result


def pca_scatter(pca_result, key, dims=(0, 1), ax=None, figsize=(4, 3.5),
                labels=('Identity', 'Non-Identity'), colors=('blue', 'red'),
                alpha=0.2, s=5):
    """
    2D matplotlib scatter of PCA projections for two activation classes.

    Args:
        pca_result (dict): Output of pca_fit with 'X_1', 'X_2', 'components'.
        key: Key into pca_result (e.g. a layer index).
        dims (tuple): Two PC indices to plot, e.g. (0, 1).
        ax: Existing matplotlib axes, or None to create a new figure.
        figsize (tuple): Figure size when ax is None.
        labels (tuple): Legend labels for X_1 and X_2.
        colors (tuple): Colors for X_1 and X_2.
        alpha (float): Point transparency.
        s (float): Point size.

    Returns:
        (fig, ax)
    """
    entry = pca_result[key]
    comp = entry['components']
    X1 = entry['X_1']
    X2 = entry.get('X_2')

    d = list(dims)
    proj1 = (X1 @ comp[d, :].T).numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if X2 is not None:
        proj2 = (X2 @ comp[d, :].T).numpy()
        ax.scatter(proj2[:, 0], proj2[:, 1], color=colors[1], alpha=alpha, s=s, label=labels[1])
    ax.scatter(proj1[:, 0], proj1[:, 1], color=colors[0], alpha=alpha, s=s, label=labels[0])

    ax.set_xlabel(f'PC{dims[0]+1}')
    ax.set_ylabel(f'PC{dims[1]+1}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best', frameon=False)
    return fig, ax


def pca_scatter_3d(pca_result, key, dims=(0, 1, 2), figsize=(5, 4),
                   labels=('Identity', 'Non-Identity'), colors=('blue', 'red'),
                   alpha=0.15, s=3):
    """
    3D matplotlib scatter of PCA projections for two activation classes.

    Args:
        pca_result (dict): Output of pca_fit with 'X_1', 'X_2', 'components'.
        key: Key into pca_result (e.g. a layer index).
        dims (tuple): Three PC indices to plot, e.g. (0, 1, 2).
        figsize (tuple): Figure size.
        labels (tuple): Legend labels for X_1 and X_2.
        colors (tuple): Colors for X_1 and X_2.
        alpha (float): Point transparency.
        s (float): Point size.

    Returns:
        (fig, ax)
    """
    entry = pca_result[key]
    comp = entry['components']
    X1 = entry['X_1']
    X2 = entry.get('X_2')

    d = list(dims)
    proj1 = (X1 @ comp[d, :].T).numpy()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    if X2 is not None:
        proj2 = (X2 @ comp[d, :].T).numpy()
        ax.scatter(proj2[:, 0], proj2[:, 1], proj2[:, 2],
                   color=colors[1], alpha=alpha, s=s, label=labels[1])
    ax.scatter(proj1[:, 0], proj1[:, 1], proj1[:, 2],
               color=colors[0], alpha=alpha, s=s, label=labels[0])

    ax.set_xlabel(f'PC{dims[0]+1}')
    ax.set_ylabel(f'PC{dims[1]+1}')
    ax.set_zlabel(f'PC{dims[2]+1}')
    ax.legend(frameon=False)
    return fig, ax

def pca_plot_plotly(pca_result, key, labels, dims=(0, 1, 2), color_by='is_identity'):
    """
    Interactive Plotly 3D scatter of PCA projections colored by a label_facts attribute.

    Designed for the token_idx=None case from collect_pca_activations, where each point
    corresponds to a token position labeled with label_facts metadata.

    Args:
        pca_result (dict): Output of pca_fit with 'X_1' and 'components'.
        key: Key into pca_result (e.g. a layer index).
        labels (list): List of label_facts objects (from collect_pca_activations with token_idx=None).
        dims (tuple): Three PC indices to visualize.
        color_by (str): label_facts.fact attribute to use for coloring (e.g. 'is_identity',
                        'is_square', 'identity_type').

    Returns:
        plotly Figure object.
    """
    import pandas as pd
    import plotly.express as px

    entry = pca_result[key]
    comp = entry['components']
    X = entry['X_1']

    d = list(dims)
    proj = (X @ comp[d, :].T).numpy()

    fact_attrs = ['is_identity', 'identity_type', 'is_square', 'square_type',
                  'seen_ab', 'seen_ba', 'is_commutative']

    df = {
        f'PC{dims[0]+1}': proj[:, 0],
        f'PC{dims[1]+1}': proj[:, 1],
        f'PC{dims[2]+1}': proj[:, 2],
    }
    for attr in fact_attrs:
        df[attr] = [str(getattr(lbl.fact, attr, None)) for lbl in labels]
    df['symbol']   = [lbl.token.symbol   for lbl in labels]
    df['slot']     = [lbl.token.slot      for lbl in labels]

    df = pd.DataFrame(df)

    hover_cols = [c for c in df.columns
                  if c not in (f'PC{dims[0]+1}', f'PC{dims[1]+1}', f'PC{dims[2]+1}', color_by)]

    fig = px.scatter_3d(
        df,
        x=f'PC{dims[0]+1}',
        y=f'PC{dims[1]+1}',
        z=f'PC{dims[2]+1}',
        color=color_by,
        color_discrete_sequence=['#636EFA', '#EF553B'],
        hover_data=hover_cols,
        opacity=0.2,
    )
    return fig
