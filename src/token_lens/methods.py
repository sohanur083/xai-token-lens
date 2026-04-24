"""Explanation methods: LIME, SHAP-ish, attention. All backend-free baselines."""
from __future__ import annotations
from typing import Callable, List, Tuple
import random, re, math


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\S+", text)


def lime_explain(
    text: str,
    predict_fn: Callable[[List[str]], List[float]],
    n_samples: int = 100,
    seed: int = 0,
) -> List[Tuple[str, float]]:
    """Simple LIME: perturb text by dropping tokens, fit a linear model.

    predict_fn takes a list of variant strings, returns scalar scores (one per variant).
    """
    rng = random.Random(seed)
    tokens = _tokenize(text)
    n = len(tokens)
    if n == 0:
        return []

    # Build perturbations as binary masks
    masks, variants = [], []
    for _ in range(n_samples):
        mask = [rng.random() > 0.35 for _ in range(n)]
        if not any(mask):
            mask[rng.randrange(n)] = True
        variant = " ".join(tok for tok, keep in zip(tokens, mask) if keep)
        masks.append(mask)
        variants.append(variant)
    masks.append([True] * n)
    variants.append(text)

    scores = predict_fn(variants)
    base = scores[-1]

    # Weighted least squares with distance = hamming to all-kept mask
    # closed-form contribution: corr(mask_i, score - base)
    contribs = []
    for i in range(n):
        xi = [1.0 if m[i] else 0.0 for m in masks]
        yi = [s - base for s in scores]
        mean_x, mean_y = sum(xi) / len(xi), sum(yi) / len(yi)
        num = sum((xi[k] - mean_x) * (yi[k] - mean_y) for k in range(len(xi)))
        den = sum((xi[k] - mean_x) ** 2 for k in range(len(xi))) or 1e-9
        contribs.append(num / den)

    return list(zip(tokens, contribs))


def shap_explain(
    text: str,
    predict_fn: Callable[[List[str]], List[float]],
    n_samples: int = 60,
    seed: int = 0,
) -> List[Tuple[str, float]]:
    """Kernel-SHAP-style: average marginal contribution via random subsets."""
    rng = random.Random(seed)
    tokens = _tokenize(text)
    n = len(tokens)
    if n == 0:
        return []

    contribs = [0.0] * n
    counts = [0] * n

    for _ in range(n_samples):
        order = list(range(n))
        rng.shuffle(order)
        present = [False] * n
        variants, marginals = [], []
        for idx in order:
            before = " ".join(tokens[j] for j in range(n) if present[j]) or "."
            present[idx] = True
            after = " ".join(tokens[j] for j in range(n) if present[j])
            variants.append((idx, before, after))
        flat_inputs = [v for _, b, a in variants for v in (b, a)]
        scores = predict_fn(flat_inputs)
        for k, (idx, _, _) in enumerate(variants):
            before_s, after_s = scores[2 * k], scores[2 * k + 1]
            contribs[idx] += after_s - before_s
            counts[idx] += 1

    return [(tokens[i], contribs[i] / max(counts[i], 1)) for i in range(n)]


def attention_explain(
    text: str,
    attention_weights: List[List[float]] | None = None,
) -> List[Tuple[str, float]]:
    """If attention_weights[i][j] is provided (N x N), aggregate attention received per token."""
    tokens = _tokenize(text)
    n = len(tokens)
    if not attention_weights:
        # fallback proxy: normalized token length + position decay
        weights = [1.0 / (1 + i * 0.1) for i in range(n)]
    else:
        weights = [sum(col) / max(len(col), 1) for col in zip(*attention_weights)]
        s = sum(weights) or 1.0
        weights = [w / s for w in weights]
    return list(zip(tokens, weights))
