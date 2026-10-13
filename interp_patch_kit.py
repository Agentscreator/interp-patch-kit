"""Activation patching sweeps for transformer language models, built on TransformerLens."""

from typing import Callable, Optional, Tuple

import torch
from transformer_lens.model_bridge import TransformerBridge

# Hook points that can be patched. "z" is per-head; the rest patch a whole layer.
HOOKS = {
    "z": "blocks.{layer}.attn.hook_z",
    "attn_out": "blocks.{layer}.hook_attn_out",
    "mlp_out": "blocks.{layer}.hook_mlp_out",
    "resid_pre": "blocks.{layer}.hook_resid_pre",
    "resid_post": "blocks.{layer}.hook_resid_post",
}
PER_HEAD_HOOKS = {"z"}


def load_model(model_name: str, device: Optional[str] = None) -> TransformerBridge:
    """Load a TransformerBridge model with compatibility mode enabled."""
    kwargs = {"device": device} if device is not None else {}
    model = TransformerBridge.boot_transformers(model_name, **kwargs)
    model.enable_compatibility_mode(disable_warnings=True)
    model.eval()
    return model


def patch_sweep(
    model: TransformerBridge,
    clean_prompt: str,
    corrupted_prompt: str,
    n_layers: int,
    n_heads: int,
    score_fn: Callable[[torch.Tensor], float],
    hook: str = "z",
) -> torch.Tensor:
    """Patch clean activations into a corrupted run, one component at a time.

    For each layer (and each head, if ``hook`` is per-head), the clean activation
    at that hook point replaces the corrupted one, and ``score_fn(logits)`` is
    recorded.

    Args:
        model: A loaded model, e.g. from ``load_model``.
        clean_prompt: Prompt whose activations are patched in.
        corrupted_prompt: Prompt that is run with the patch applied. Must tokenize
            to the same length as ``clean_prompt``.
        n_layers: Number of layers to sweep.
        n_heads: Number of attention heads per layer.
        score_fn: Takes logits of shape ``[batch, pos, d_vocab]`` and returns a
            single number for how strongly the desired answer is predicted.
        hook: Which hook point to patch. One of ``HOOKS``. ``"z"`` (attention
            head output) patches each head separately; the others patch a whole
            layer, so the result has a single column.

    Returns:
        A ``[n_layers, n_heads]`` tensor of scores, or ``[n_layers, 1]`` for
        whole-layer hooks.
    """
    if hook not in HOOKS:
        raise ValueError(f"Unknown hook {hook!r}; choose one of {sorted(HOOKS)}")

    clean_tokens = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)
    if clean_tokens.shape != corrupted_tokens.shape:
        raise ValueError(
            f"Clean and corrupted prompts must have the same number of tokens "
            f"(got {clean_tokens.shape[-1]} and {corrupted_tokens.shape[-1]})"
        )

    hook_names = [HOOKS[hook].format(layer=layer) for layer in range(n_layers)]
    per_head = hook in PER_HEAD_HOOKS

    with torch.no_grad():
        _, clean_cache = model.run_with_cache(
            clean_tokens, names_filter=lambda name: name in hook_names
        )

        grid = torch.zeros(n_layers, n_heads if per_head else 1)
        for layer, hook_name in enumerate(hook_names):
            clean_act = clean_cache[hook_name]
            for head in range(grid.shape[1]):

                def patch(activation, hook, head=head, clean_act=clean_act):
                    # hook_z is [batch, pos, head, d_head]; other hooks have no head dim.
                    if per_head:
                        activation[:, :, head] = clean_act[:, :, head]
                    else:
                        activation[:] = clean_act
                    return activation

                logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[(hook_name, patch)])
                grid[layer, head] = float(score_fn(logits))

    return grid


def plot_heatmap(
    grid: torch.Tensor,
    title: str = "Activation patching",
    save_path: Optional[str] = None,
    show_best: bool = True,
) -> Tuple[int, int]:
    """Plot a layers-by-heads score grid and return the best ``(layer, head)``.

    The figure is saved to ``save_path`` if given, otherwise shown. If
    ``show_best`` is set, the best combination is printed as well.
    """
    import matplotlib.pyplot as plt

    grid = torch.as_tensor(grid).detach().cpu()
    n_layers, n_heads = grid.shape

    fig, ax = plt.subplots(figsize=(max(4, n_heads * 0.6 + 2), max(4, n_layers * 0.45 + 1)))
    image = ax.imshow(grid.numpy(), cmap="RdBu_r", aspect="auto", origin="lower")
    fig.colorbar(image, ax=ax, label="Score")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

    best_layer, best_head = divmod(int(grid.argmax()), n_heads)
    if show_best:
        print(f"Best: layer {best_layer}, head {best_head} (score {grid[best_layer, best_head]:.4f})")
    return best_layer, best_head
