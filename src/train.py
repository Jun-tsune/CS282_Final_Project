import os
from random import randint
import uuid
from tqdm import tqdm
import torch
from eval import get_run_metrics
from tasks import get_task_sampler
from src.utils.samplers import get_data_sampler
from src.utils.utils_train import Curriculum
from src.utils.utils_model import build_model
import wandb
from omegaconf import OmegaConf
import time

torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func, recon_weight, memory_state=None):
    optimizer.zero_grad()

    if memory_state is not None:
        output = model(xs, ys, memories=memory_state)
    else:
        output = model(xs, ys)
    
    new_memory_state = memory_state
    
    if isinstance(output, tuple):
        # Compressive Transformer case
        y_pred, new_memory_state, aux_loss = output
        task_loss = loss_func(y_pred, ys)
        loss = task_loss + recon_weight * (aux_loss if isinstance(aux_loss, torch.Tensor) else 0.0)
    else:
        # Normal Transformer case
        y_pred = output
        loss = loss_func(y_pred, ys)

    loss.backward()
    optimizer.step()
    return loss.detach().item(), y_pred.detach(), new_memory_state


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    # --------- 1. Device selection & CUDA diagnostics ---------
    if args.training.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"Using device: CUDA ({device_name})")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        print(f"Using device: CPU")
        if args.training.device == "cuda":
            print(f"Warning: CUDA requested but not available, falling back to CPU")
            print(f"\nCUDA Diagnostics:")
            print(f"  PyTorch version: {torch.__version__}")
            print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
            try:
                if hasattr(torch.version, "cuda") and torch.version.cuda is not None:
                    print(f"  PyTorch CUDA version (compiled): {torch.version.cuda}")
                    try:
                        if torch.backends.cudnn.is_available():
                            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
                        else:
                            print(f"  cuDNN version: N/A (not available)")
                    except Exception:
                        print(f"  cuDNN version: N/A (error checking)")
                else:
                    print(f"  PyTorch CUDA version (compiled): None (CPU-only build)")
            except Exception as e:
                print(f"  Error checking CUDA info: {e}")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    try:
        recon_weight = float(getattr(args.model, "reconstruction_loss_weight", 0.0))
    except Exception:
        recon_weight = 0.0

    # --------- 2. Resume training if checkpoint exists ---------
    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for _ in range(state["train_step"] + 1):
            curriculum.update()
        print(f"Resumed training from step {starting_step}")

    # Start measuring CUDA peak memory usage from here
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples
    memory_state = None

    # --------- 3. Variables for throughput & OOM measurement ---------
    tokens_seen = 0
    last_log_time = time.time()
    last_log_tokens = 0
    oom_flag = 0 

    try:
        for i in pbar:
            data_sampler_args = {}
            task_sampler_args = {}

            # For sparse tasks, only sample valid coords
            if "sparse" in args.training.task:
                task_sampler_args["valid_coords"] = curriculum.n_dims_truncated

            # For fixed pools, sample seeds for both task and input data
            if num_training_examples is not None:
                assert num_training_examples >= bsize
                seeds = sample_seeds(num_training_examples, bsize)
                data_sampler_args["seeds"] = seeds
                task_sampler_args["seeds"] = [s + 1 for s in seeds]

            xs = data_sampler.sample_xs(
                curriculum.n_points,
                bsize,
                curriculum.n_dims_truncated,
                **data_sampler_args,
            )
            task = task_sampler(**task_sampler_args)
            ys = task.evaluate(xs)

            loss_func = task.get_training_metric()

            loss, output, memory_state = train_step(
                model,
                xs.to(device),
                ys.to(device),
                optimizer,
                loss_func,
                recon_weight,
                memory_state=None,
            )

            # Pointwise losses (for debugging or fine-grained metrics)
            point_wise_tags = list(range(curriculum.n_points))
            point_wise_loss_func = task.get_metric()
            point_wise_loss = point_wise_loss_func(
                output, ys.to(device)
            ).mean(dim=0)

            # Baseline (random guess or trivial baseline depending on dims)
            baseline_loss = (
                sum(
                    max(curriculum.n_dims_truncated - ii, 0)
                    for ii in range(curriculum.n_points)
                )
                / curriculum.n_points
            )

            # Count tokens processed in this step (for throughput)
            tokens_this_step = curriculum.n_points * bsize
            tokens_seen += tokens_this_step

            # --------- 4. W&B logging ---------
            if i % args.wandb.log_every_steps == 0 and not args.test_run:
                if wandb.run is not None:
                    try:
                        # Compute throughput
                        now = time.time()
                        dt = max(now - last_log_time, 1e-8)
                        tokens_since_last = tokens_seen - last_log_tokens

                        throughput_tokens_per_s = tokens_since_last / dt
                        steps_per_s = args.wandb.log_every_steps / dt

                        # CUDA peak memory (GB, running peak)
                        gpu_peak_mem_gb = None
                        if torch.cuda.is_available():
                            gpu_peak_mem_gb = (
                                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                            )

                        log_dict = {
                            "global_step": i,
                            # --- task performance ---
                            "train/overall_loss": float(loss),
                            "train/excess_loss": float(loss / baseline_loss),

                            # "train/pointwise_loss": dict(
                            #     zip(point_wise_tags, point_wise_loss.cpu().numpy())
                            # ),

                            # --- sequence & model config ---
                            "config/seq_len": curriculum.n_points*2,
                            "config/n_dims_truncated": curriculum.n_dims_truncated,
                            "config/batch_size": bsize,
                            "config/chunk_size": getattr(args.model, "chunk_size", None),
                            "config/mem_len": getattr(args.model, "mem_len", None),
                            "config/cmem_ratio": getattr(args.model, "cmem_ratio", None),
                            "config/arch": getattr(args.model, "model_family", "unknown"),

                            # --- throughput ---
                            "throughput/tokens_per_s": throughput_tokens_per_s,
                            "throughput/steps_per_s": steps_per_s,

                            # --- GPU memory (running peak) ---
                            "gpu/peak_mem_GB": gpu_peak_mem_gb,
                        }

                        wandb.log(log_dict, step=i)

                        last_log_time = now
                        last_log_tokens = tokens_seen

                    except Exception:
                        # Do not interrupt training if W&B logging fails
                        pass

            curriculum.update()

            pbar.set_description(f"loss {loss:.6f}")

            # Save checkpoint
            if i % args.training.save_every_steps == 0 and not args.test_run:
                training_state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_step": i,
                }
                torch.save(training_state, state_path)

            # Periodically save full model snapshots
            if (
                args.training.keep_every_steps > 0
                and i % args.training.keep_every_steps == 0
                and not args.test_run
                and i > 0
            ):
                torch.save(
                    model.state_dict(),
                    os.path.join(args.out_dir, f"model_{i}.pt"),
                )

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("⚠️  Caught CUDA out-of-memory error during training.")
            oom_flag = 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            raise

    gpu_max_memory_mb = None
    if torch.cuda.is_available():
        max_bytes = torch.cuda.max_memory_allocated(device)
        gpu_max_memory_mb = max_bytes / (1024 ** 2)
        print(f"[train] Peak GPU memory: {gpu_max_memory_mb:.1f} MB")

    print(f"[train] args.test_run = {args.test_run}, wandb.run is None? {wandb.run is None}")

    if wandb.run is not None and not args.test_run:
        try:
            wandb.summary["gpu_max_memory_mb"] = gpu_max_memory_mb
            wandb.summary["oom_flag"] = oom_flag
            wandb.summary["device_name"] = device_name
            print("[train] Wrote gpu_max_memory_mb / oom_flag / device_name to wandb.summary")
        except Exception as e:
            print(f"[train] Failed to write wandb.summary: {e}")


def make_wandb_config(args):
    m = args.model

    def mget(name, default=None):
        return getattr(m, name, default)

    cfg = {
        "out_dir": args.out_dir,
        "train_steps": args.training.train_steps,
        "batch_size": args.training.batch_size,
        "learning_rate": args.training.learning_rate,
        "task": args.training.task,
        "data": args.training.data,
        "curriculum": {
            "dims": {
                "start": args.training.curriculum.dims.start,
                "end": args.training.curriculum.dims.end,
            },
            "points": {
                "start": args.training.curriculum.points.start,
                "end": args.training.curriculum.points.end,
            },
        },
        "model": {
            "model_family": mget("model_family"),
            "n_embd": mget("n_embd"),
            "n_layer": mget("n_layer"),
            "n_head": mget("n_head"),
            "n_dims": mget("n_dims"),
            "n_positions": mget("n_positions"),
            "embd_pdrop": mget("embd_pdrop"),
            "resid_pdrop": mget("resid_pdrop"),
            "attn_pdrop": mget("attn_pdrop"),
            "mem_len": mget("mem_len"),
            "cmem_ratio": mget("cmem_ratio"),
            "cmem_len": mget("cmem_len"),
            "recon_attn_dropout": mget("recon_attn_dropout"),
            "reconstruction_loss_weight": mget("reconstruction_loss_weight"),
        },
    }
    return cfg



def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        # Try to initialize wandb, fallback to offline mode if API key is not configured
        run_name = None
        if "wandb" in args:
            run_name = args.wandb.get("name", None)
        if wandb.run is None:
            try:
                cfg = make_wandb_config(args)
                wandb.init(
                    dir=args.out_dir,
                    project=args.wandb.project,
                    entity=args.wandb.entity,
                    config=cfg,
                    notes=args.wandb.notes,
                    name=run_name,
                    resume=True,
                )
            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}")
                print("Falling back to offline mode...")
                try:
                    if wandb.run is not None:
                        wandb.finish()
                    cfg = make_wandb_config(args)
                    wandb.init(
                        dir=args.out_dir,
                        project=args.wandb.project,
                        entity=args.wandb.entity,
                        config=cfg,
                        notes=args.wandb.notes,
                        name=run_name,
                        resume=True,
                        mode="offline",
                    )
                    print("Wandb initialized in offline mode. Run 'wandb sync' later to upload.")
                except Exception as e2:
                    print(f"Warning: Failed to initialize wandb even in offline mode: {e2}")
                    print("Continuing training without wandb logging...")
        else:
            wandb.log({"already_running": 1})

    model = build_model(args.model)
    model.train()

    train(model, args)
    
    # Clean up GPU memory after training
    if torch.cuda.is_available():
        import gc
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # if not args.test_run:
    #     _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    default_config = {
        "out_dir": "outputs",
        "model_yaml": "config_model_1",
        "train_yaml": "config_train_1",
        "test_run": True,
    }
    cfg = OmegaConf.create(default_config)

    # Receive command line arguments and merge with config
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Load configuration files
    cfg.model_yaml = os.path.join("src/config/config_model/", cfg.model_yaml + ".yaml")
    cfg.train_yaml = os.path.join("src/config/config_train/", cfg.train_yaml + ".yaml")
    cfg_model = OmegaConf.load(cfg.model_yaml)
    cfg_train = OmegaConf.load(cfg.train_yaml)

    # Load standard config which is not changed frequently
    cfg_standard = OmegaConf.load(os.path.join("src/config/", "standard.yaml"))

    # Merge all configurations
    args = OmegaConf.merge(cfg, cfg_standard, cfg_model, cfg_train, cli_cfg)

    # Create output directory and save final config
    run_id = args.training.resume_id
    out_dir = os.path.join(args.out_dir, 'id_' + str(run_id))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    args.out_dir = out_dir
    OmegaConf.save(args, os.path.join(out_dir, "config.yaml"))

    print(f"Running with: {args}")

    main(args)
