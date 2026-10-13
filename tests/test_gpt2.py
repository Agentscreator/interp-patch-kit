"""End-to-end check on GPT-2 Small. Run with: python -m pytest tests/"""

import pytest
import torch

from interp_patch_kit import load_model, patch_sweep, plot_heatmap

CLEAN = "When John and Mary went to the store, John gave a drink to"
CORRUPTED = "When John and Mary went to the store, Mary gave a drink to"


@pytest.fixture(scope="module")
def model():
    return load_model("gpt2")


@pytest.fixture(scope="module")
def score_fn(model):
    mary = model.to_single_token(" Mary")
    john = model.to_single_token(" John")
    return lambda logits: (logits[0, -1, mary] - logits[0, -1, john]).item()


def test_head_sweep(model, score_fn, tmp_path):
    grid = patch_sweep(model, CLEAN, CORRUPTED, model.cfg.n_layers, model.cfg.n_heads, score_fn)
    assert grid.shape == (12, 12)

    clean_score = score_fn(model(CLEAN))
    corrupted_score = score_fn(model(CORRUPTED))
    assert clean_score > 0 > corrupted_score
    # Some single head should move the score well toward the clean answer.
    assert grid.max() > corrupted_score + 0.1 * (clean_score - corrupted_score)

    out = tmp_path / "heatmap.png"
    layer, head = plot_heatmap(grid, save_path=str(out))
    assert out.exists()
    assert grid[layer, head] == grid.max()


def test_full_residual_patch_restores_clean(model, score_fn):
    grid = patch_sweep(
        model, CLEAN, CORRUPTED, 1, model.cfg.n_heads, score_fn, hook="resid_pre"
    )
    assert grid.shape == (1, 1)
    assert grid[0, 0] == pytest.approx(score_fn(model(CLEAN)), abs=1e-3)


def test_length_mismatch_raises(model, score_fn):
    with pytest.raises(ValueError, match="same number of tokens"):
        patch_sweep(model, CLEAN, "Short prompt", 12, 12, score_fn)


def test_unknown_hook_raises(model, score_fn):
    with pytest.raises(ValueError, match="Unknown hook"):
        patch_sweep(model, CLEAN, CORRUPTED, 12, 12, score_fn, hook="nope")
