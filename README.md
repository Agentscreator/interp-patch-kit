# interp-patch-kit

Activation patching sweeps for transformer language models, built on
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) 4's `TransformerBridge`.

Give it a clean prompt, a corrupted prompt and a scoring function. For every layer and head, it
patches that head's clean activation into a corrupted run and records the score.

![IOI patching heatmap on GPT-2 Small](example_heatmap.png)

## Install

```bash
pip install -e .            # add [test] for pytest
```

## Usage

```python
from interp_patch_kit import load_model, patch_sweep, plot_heatmap

model = load_model("gpt2")  # TransformerBridge with compatibility mode enabled

mary, john = model.to_single_token(" Mary"), model.to_single_token(" John")
score = lambda logits: (logits[0, -1, mary] - logits[0, -1, john]).item()

grid = patch_sweep(
    model,
    clean_prompt="When John and Mary went to the store, John gave a drink to",
    corrupted_prompt="When John and Mary went to the store, Mary gave a drink to",
    n_layers=model.cfg.n_layers,
    n_heads=model.cfg.n_heads,
    score_fn=score,
)
layer, head = plot_heatmap(grid, save_path="heatmap.png")  # prints and returns the best head
```

## API

- **`load_model(model_name, device=None)`** returns a `TransformerBridge` with compatibility mode
  already enabled.
- **`patch_sweep(model, clean_prompt, corrupted_prompt, n_layers, n_heads, score_fn, hook="z")`**
  returns a `[n_layers, n_heads]` grid of scores. `score_fn` takes logits of shape
  `[batch, pos, d_vocab]` and returns one number. The two prompts must tokenize to the same length.
- **`plot_heatmap(grid, title=..., save_path=None, show_best=True)`** draws heads against layers,
  saves the figure (or shows it if no path is given) and returns the best `(layer, head)`.

### Hook points

| `hook`       | Patches                                | Grid shape          |
|--------------|----------------------------------------|---------------------|
| `z` (default)| Each attention head's output (`attn.hook_z`) | `[n_layers, n_heads]` |
| `attn_out`   | Whole attention layer output           | `[n_layers, 1]`     |
| `mlp_out`    | Whole MLP output                       | `[n_layers, 1]`     |
| `resid_pre`  | Residual stream entering the block     | `[n_layers, 1]`     |
| `resid_post` | Residual stream leaving the block      | `[n_layers, 1]`     |

Whole-layer hooks patch every position at once. With `resid_pre`, every layer therefore restores
the full clean run, so it is mostly useful as a sanity check.

## Tests

```bash
python -m pytest tests/   # downloads GPT-2 Small on first run
```
