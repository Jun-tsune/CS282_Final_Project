# -*- coding: utf-8 -*-
import torch


@torch.no_grad()
def get_regression_batch(
    batch_size, d, k_context, q_queries=1, noise_std=0.0, device="cpu"
):
    """
    Generate a batch of scalar linear regression ICL episodes.
    Each episode uses y = x^T w + eps.

    Returns:
      x_ctx: [B, K, d]
      y_ctx: [B, K, 1]
      x_q  : [B, Q, d]
      y_q  : [B, Q, 1]
    """
    B, K, Q = batch_size, k_context, q_queries
    g = torch.Generator(device=device)

    # Task parameter w ~ N(0, I/d) per episode
    w = torch.randn(B, d, generator=g, device=device) / (d**0.5)  # [B, d]

    # Context pairs
    x_ctx = torch.randn(B, K, d, generator=g, device=device)
    y_ctx = (x_ctx * w[:, None, :]).sum(-1, keepdim=True)
    if noise_std > 0:
        y_ctx = y_ctx + noise_std * torch.randn_like(y_ctx, generator=g)

    # Query pairs
    x_q = torch.randn(B, Q, d, generator=g, device=device)
    y_q = (x_q * w[:, None, :]).sum(-1, keepdim=True)
    if noise_std > 0:
        y_q = y_q + noise_std * torch.randn_like(y_q, generator=g)

    return x_ctx, y_ctx, x_q, y_q
