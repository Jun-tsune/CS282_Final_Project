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

torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func, recon_weight):
    optimizer.zero_grad()
    output = model(xs, ys)
    if isinstance(output, tuple):
        # Compressive Transformer case
        y_pred, memory, aux_loss = output
        task_loss = loss_func(y_pred, ys)
        loss = task_loss + recon_weight * (aux_loss if isinstance(aux_loss, torch.Tensor) else 0.0)
    else:
        # Normal Transformer case
        y_pred = output
        loss = loss_func(y_pred, ys)

    loss.backward()
    optimizer.step()
    return loss.detach().item(), y_pred.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    device = torch.device(args.training.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)
    
    try:
        recon_weight = float(getattr(args.model, "reconstruction_loss_weight", 0.0))
    except Exception:
        recon_weight = 0.0

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

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

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
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

        loss, output = train_step(model, xs.to(device), ys.to(device), optimizer, loss_func, recon_weight)

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.to(device)).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )
        wandb.log({"args": args})

    model = build_model(args.model)
    model.train()

    train(model, args)

    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


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
    args = OmegaConf.merge(cfg, cfg_standard, cfg_model, cfg_train)

    # Create output directory and save final config
    run_id = args.training.resume_id
    if run_id is None:
        run_id = str(uuid.uuid4())
    out_dir = os.path.join(args.out_dir, 'id_' + str(run_id))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    args.out_dir = out_dir
    OmegaConf.save(args, os.path.join(out_dir, "config.yaml"))

    print(f"Running with: {args}")

    main(args)
