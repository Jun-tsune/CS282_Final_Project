import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_generator import get_regression_batch
from model_standard import StandardTransformer  # Chage later
from model_placeholder import CompressiveTransformer  # Chage later


def parse_int_list(s):
    return [int(x) for x in s.split(",")] if s else []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=20)
    p.add_argument("--train_k", type=int, default=128)  # train context length
    p.add_argument(
        "--eval_ks", type=str, default="128,256,512,1024,2048"
    )  # comma-separated
    p.add_argument("--q_queries", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--noise_std", type=float, default=0.0)
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument(
        "--model", type=str, choices=["standard", "compressive"], default="standard"
    )
    # placeholders for future compressive settings
    p.add_argument("--memory_ratio", type=str, default="1:1")  # e.g., 1:1,1:4,1:8
    p.add_argument("--compression", type=str, default="avgpool")  # placeholder
    args = p.parse_args()

    device = torch.device(args.device)
    eval_ks = parse_int_list(args.eval_ks)

    # model select
    if args.model == "standard":
        model = StandardTransformer(d=args.d).to(device)
    else:
        # later: replace with real CompressiveTransformer(**parsed_ratio, compression=...)
        model = CompressiveTransformer(d=args.d).to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # ---------------- Train on train_k
    model.train()
    for step in range(1, args.steps + 1):
        x_ctx, y_ctx, x_q, y_q = get_regression_batch(
            batch_size=args.batch_size,
            d=args.d,
            k_context=args.train_k,
            q_queries=args.q_queries,
            noise_std=args.noise_std,
            device=device,
        )
        y_hat = model(x_ctx, y_ctx, x_q)
        loss = loss_fn(y_hat, y_q)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == args.steps:
            print(f"step {step}/{args.steps}  loss {loss.item():.6f}")

    # ---------------- Evaluate across lengths (cross-length + scaling)
    model.eval()
    with torch.no_grad():
        for k in eval_ks:
            x_ctx, y_ctx, x_q, y_q = get_regression_batch(
                batch_size=256,
                d=args.d,
                k_context=k,
                q_queries=args.q_queries,
                noise_std=args.noise_std,
                device=device,
            )
            y_hat = model(x_ctx, y_ctx, x_q)
            val_loss = loss_fn(y_hat, y_q).item()
            print(f"[eval] K={k}  val_mse {val_loss:.6f}")
