"""Train a GAN or a diffusion model.

The file reads top to bottom:

1. data          - DataLoader and an endless stream of batches
2. checkpoints   - save / load everything needed to resume
3. logging       - metrics.jsonl (+ TensorBoard), preview grids
4. train()       - the loop (takes ready-made objects, easy to test)
5. CLI           - main() builds everything from a preset and flags

Run it with:  imagegen-train --preset mnist-dcgan          (imagegen-train --help for all flags)
Multi-GPU:    torchrun --nproc_per_node=4 -m imagegen.train --preset cifar10-resnet
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import random
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from imagegen.config import (
    PRESETS,
    DataConfig,
    ModelConfig,
    TrainConfig,
    config_to_dict,
)
from imagegen.images import save_grid
from imagegen.runtime import (
    DistInfo,
    all_reduce_mean,
    get_dist_info,
    local_main_first,
    resolve_device,
    resolve_precision,
    seed_everything,
    setup_distributed,
)
from imagegen.tasks import Task, build_task

try:  # numpy is optional; we only need it for RNG snapshots
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

logger = logging.getLogger(__name__)

LAST_CHECKPOINT = "last.pt"
BEST_CHECKPOINT = "best.pt"  # lowest FID so far (only written when --fid-every > 0)

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------


def build_loader(dataset: Dataset, config: TrainConfig, device: torch.device) -> DataLoader:
    """Shuffled, full batches only. With several processes each gets its own slice."""
    info = get_dist_info()
    if len(dataset) < config.batch_size * info.world_size:
        raise ValueError(
            f"dataset has {len(dataset)} images, fewer than one batch "
            f"({config.batch_size} x {info.world_size} processes)"
        )
    workers = config.num_workers
    generator = torch.Generator().manual_seed(config.seed + info.rank)
    sampler = (
        DistributedSampler(dataset, shuffle=True, seed=config.seed, drop_last=True)
        if info.enabled
        else None
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,  # every step sees a full batch
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
    )


def endless(
    loader: DataLoader, start_epoch: int = 0
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Batches forever, reshuffling at every epoch.

    GANs and diffusion models are trained for a number of *steps*, not epochs,
    and a GAN step may consume several batches (one per discriminator update).
    """
    epoch = start_epoch
    while True:
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


# ---------------------------------------------------------------------------
# 2. Checkpoints
# ---------------------------------------------------------------------------


@dataclass
class TrainState:
    """Counters that must survive a restart."""

    step: int = 0
    best_fid: float = math.inf
    epoch: int = 0  # passes over the data so far (approximate for GANs, d_steps > 1)


def rng_state() -> dict[str, Any]:
    """Snapshot every RNG so a resumed run continues the same random stream."""
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if np is not None:
        name, keys, pos, has_gauss, cached = np.random.get_state()
        state["numpy"] = (name, keys.tolist(), pos, has_gauss, cached)  # weights_only-safe
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if np is not None and "numpy" in state:
        name, keys, pos, has_gauss, cached = state["numpy"]
        np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    task: Task,
    state: TrainState,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    class_names: list[str],
) -> None:
    """Everything needed to resume or to sample. Written atomically."""
    checkpoint = {
        "task": task.state_dict(),
        "step": state.step,
        "epoch": state.epoch,
        "best_fid": state.best_fid,
        "data_config": config_to_dict(data_config),
        "model_config": config_to_dict(model_config),
        "train_config": config_to_dict(train_config),
        "class_names": class_names,
        "rng_state": rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(path)  # atomic: a crash mid-save never leaves a half-written file


def load_checkpoint(path: Path, task: Task) -> TrainState:
    """Load weights + optimizer states into `task`; return the saved counters."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    task.load_state_dict(checkpoint["task"])
    set_rng_state(checkpoint["rng_state"])
    return TrainState(checkpoint["step"], checkpoint["best_fid"], checkpoint.get("epoch", 0))


# ---------------------------------------------------------------------------
# 3. Logging
# ---------------------------------------------------------------------------


class MetricsLog:
    """Appends one JSON object per event to metrics.jsonl (and TensorBoard if asked).

    A fresh run (start_step 0) starts a new file; a resumed run keeps the
    events up to start_step and drops later ones (written after that checkpoint).
    """

    def __init__(self, output_dir: Path, tensorboard: bool = False, start_step: int = 0) -> None:
        self.path = output_dir / "metrics.jsonl"
        kept = []
        if start_step > 0 and self.path.exists():
            kept = [
                line for line in self.path.read_text().splitlines()
                if line and json.loads(line)["step"] <= start_step
            ]  # fmt: skip
        self.path.write_text("".join(line + "\n" for line in kept))
        self.writer = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(output_dir / "tensorboard")

    def write(self, event: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(event) + "\n")
        if self.writer is not None:
            for key, value in event.items():
                if isinstance(value, (int, float)) and key not in ("step", "epoch"):
                    self.writer.add_scalar(f"{event['kind']}/{key}", value, event["step"])

    def add_image(self, tag: str, grid_path: Path, step: int) -> None:
        if self.writer is not None:
            import numpy as np  # tensorboard depends on numpy anyway
            from PIL import Image

            image = np.asarray(Image.open(grid_path))
            self.writer.add_image(tag, image, step, dataformats="HWC")

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def preview_inputs(task: Task, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fixed noise (and labels) for the preview grids, so grids are comparable over time.

    For a conditional model, row r of the grid shows class r % num_classes.
    """
    generator = torch.Generator().manual_seed(seed)
    noise = task.noise(n, generator)
    labels = None
    if task.conditional:
        per_row = max(1, int(math.sqrt(n)))
        labels = (torch.arange(n) // per_row % task.model_config.num_classes).to(task.device)
    return noise, labels


# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------


def out_of_time_budget(start_time: float, budget_hours: float, device: torch.device) -> bool:
    """True once `budget_hours` have passed. All processes agree (rank 0 decides)."""
    if budget_hours <= 0:
        return False
    flag = torch.tensor(
        float((time.perf_counter() - start_time) / 3600 > budget_hours), device=device
    )
    if get_dist_info().enabled:
        dist.broadcast(flag, src=0)  # clocks differ slightly: never let ranks disagree
    return bool(flag.item())


def train(
    task: Task,
    loader: DataLoader,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    class_names: list[str],
    callback: Callable[[dict[str, Any]], None] | None = None,
    fid_fn: Callable[[Task], float] | None = None,
) -> TrainState:
    """Run training and return the final counters.

    Everything is passed in already built, so tests can use tiny models.
    `callback` receives every logged event (main process only); the notebook
    uses it for its live dashboard:
        {"kind": "train", "step", "epoch", "total_steps", "<metric>": value, ..., "it_per_sec"}
        {"kind": "sample", "step", "path"}
        {"kind": "fid", "step", "fid"}
    `fid_fn(task) -> fid` is called every fid_every steps on every process.
    """
    cfg = train_config
    info = get_dist_info()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = TrainState()
    if cfg.resume is not None:
        state = load_checkpoint(Path(cfg.resume), task)
        logger.info("resumed from %s at step %d", cfg.resume, state.step)

    log = MetricsLog(output_dir, cfg.tensorboard, state.step) if info.is_main else None
    preview_noise, preview_labels = preview_inputs(task, cfg.num_samples, cfg.seed)
    batches_per_epoch = len(loader)
    batches = endless(loader, start_epoch=state.epoch)

    def emit(event: dict[str, Any]) -> None:
        if info.is_main:
            log.write(event)
            if callback is not None:
                callback(event)

    def checkpoint(path: Path) -> None:
        if info.is_main:  # one writer: all processes hold the same weights
            save_checkpoint(
                path, task, state, data_config, model_config, cfg, class_names
            )  # fmt: skip

    def save_preview() -> None:
        if not info.is_main:
            return
        images = task.generate(
            preview_noise, preview_labels, steps=cfg.sample_steps,
            guidance_scale=cfg.guidance_scale,
        )  # fmt: skip
        path = save_grid(images, output_dir / "samples" / f"step_{state.step:07d}.png")
        log.add_image("samples", path, state.step)
        emit({"kind": "sample", "step": state.step, "path": str(path)})

    logger.info(
        "training %s (%s) for %d steps, batch %d x %d processes, on %s, amp=%s",
        model_config.arch, task.family, cfg.steps, cfg.batch_size, info.world_size,
        task.device, task.amp_dtype,
    )  # fmt: skip
    if info.is_main and state.step == 0:
        save_grid(next(iter(loader))[0][: cfg.num_samples], output_dir / "real.png")

    sums: dict[str, torch.Tensor] = {}
    start_time = window_start = time.perf_counter()
    window_steps = 0
    batches_used = 0
    batches_per_step = cfg.d_steps if task.family == "gan" else 1

    while state.step < cfg.steps:
        metrics = task.train_step(batches, state.step)
        state.step += 1
        window_steps += 1
        batches_used += batches_per_step
        if batches_used >= batches_per_epoch:
            state.epoch += 1
            batches_used -= batches_per_epoch
        for key, value in metrics.items():  # stays on device: no GPU sync every step
            sums[key] = sums.get(key, 0.0) + value

        if state.step % cfg.log_every == 0 or state.step == cfg.steps:
            elapsed = time.perf_counter() - window_start
            means = {k: float(v) / window_steps for k, v in sums.items()}
            means = all_reduce_mean(means, task.device)
            if not all(math.isfinite(v) for v in means.values()):
                raise FloatingPointError(f"non-finite metrics at step {state.step}: {means}")
            emit({
                "kind": "train", "step": state.step, "epoch": state.epoch,
                "total_steps": cfg.steps, **means, "it_per_sec": window_steps / elapsed,
            })  # fmt: skip
            logger.info(
                "step %d/%d | %s | %.1f it/s",
                state.step, cfg.steps, " | ".join(f"{k} {v:.4f}" for k, v in means.items()),
                window_steps / elapsed,
            )  # fmt: skip
            sums, window_steps = {}, 0
            window_start = time.perf_counter()

        if state.step % cfg.sample_every == 0 or state.step == cfg.steps:
            save_preview()

        if (
            fid_fn is not None
            and cfg.fid_every > 0
            and (state.step % cfg.fid_every == 0 or state.step == cfg.steps)
        ):
            fid = fid_fn(task)
            emit({"kind": "fid", "step": state.step, "fid": fid})
            logger.info("fid | step %d | %.2f", state.step, fid)
            if fid < state.best_fid:
                state.best_fid = fid
                checkpoint(output_dir / BEST_CHECKPOINT)

        out_of_time = state.step % cfg.log_every == 0 and out_of_time_budget(
            start_time, cfg.time_budget_hours, task.device
        )
        if state.step % cfg.ckpt_every == 0 or state.step == cfg.steps or out_of_time:
            checkpoint(output_dir / LAST_CHECKPOINT)
        if out_of_time:
            logger.info("time budget of %.2f h used: stopping at step %d", cfg.time_budget_hours,
                        state.step)  # fmt: skip
            break
        if state.step % cfg.log_every == 0:
            window_start = time.perf_counter()  # don't count sampling/eval time in it/s

    if log is not None:
        log.close()
    logger.info("done | %d steps | outputs in %s", state.step, output_dir)
    return state


# ---------------------------------------------------------------------------
# 5. Command line
# ---------------------------------------------------------------------------

# Fields the CLI does not expose: they are read from the dataset.
DERIVED_MODEL_FIELDS = {"image_size", "channels", "num_classes"}


def _add_field(group: argparse._ArgumentGroup, f: dataclasses.Field, default: Any) -> None:
    """One --flag per config field; the type is taken from the default value."""
    flag = "--" + f.name.replace("_", "-")
    if isinstance(default, bool):
        group.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
    elif isinstance(default, tuple):
        group.add_argument(flag, type=int, nargs="+", default=default)
    elif isinstance(default, Path) or f.name in ("data_dir", "output_dir", "resume"):
        group.add_argument(flag, type=Path, default=default)
    elif f.name == "limit":
        group.add_argument(flag, type=int, default=default)
    else:
        group.add_argument(flag, type=type(default), default=default)


def build_parser() -> argparse.ArgumentParser:
    """All CLI flags; defaults come from the config dataclasses (then the preset)."""
    p = argparse.ArgumentParser(
        description="Train a GAN (dcgan, resnet) or a diffusion model (unet).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--preset", choices=sorted(PRESETS), default=None, help="starting config")
    p.add_argument("--list-presets", action="store_true", help="print the presets and exit")
    for title, cls in (("data", DataConfig), ("model", ModelConfig), ("training", TrainConfig)):
        group = p.add_argument_group(title)
        defaults = cls()
        for f in fields(cls):
            if cls is ModelConfig and f.name in DERIVED_MODEL_FIELDS:
                continue
            _add_field(group, f, getattr(defaults, f.name))
    return p


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse twice: first find --preset, then use it as the defaults."""
    parser = build_parser()
    known, _ = parser.parse_known_args(argv)
    if known.preset is not None:
        parser.set_defaults(**PRESETS[known.preset])
    return parser.parse_args(argv)


def configs_from_args(args: argparse.Namespace) -> tuple[DataConfig, ModelConfig, TrainConfig]:
    """Parsed flags -> the three configs. num_classes is filled in once the data is loaded."""

    def pick(cls: type) -> dict[str, Any]:
        names = {f.name for f in fields(cls)}
        if cls is ModelConfig:
            names -= DERIVED_MODEL_FIELDS
        return {
            k: (tuple(v) if isinstance(v, list) else v) for k, v in vars(args).items() if k in names
        }

    data = DataConfig(**pick(DataConfig))
    model = ModelConfig(**pick(ModelConfig), image_size=data.image_size, channels=data.channels)
    return data, model, TrainConfig(**pick(TrainConfig))


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point. Returns None (console scripts pass the return to sys.exit)."""
    args = parse_args(argv)
    if args.list_presets:
        for name, preset in PRESETS.items():
            print(f"{name:20s} arch={preset['arch']:7s} dataset={preset['dataset']}")
        return
    data_config, model_config, train_config = configs_from_args(args)
    info = setup_distributed(train_config.device)
    logging.basicConfig(
        level=logging.INFO if info.is_main else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        build_and_train(data_config, model_config, train_config, info)
    finally:
        if info.enabled:
            dist.destroy_process_group()


def build_and_train(
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
    info: DistInfo | None = None,
    callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Task, TrainState]:
    """Dataset -> model -> train. Shared by the CLI and the notebook."""
    from imagegen.data import build_dataset, dataset_classes, num_classes
    from imagegen.models import count_parameters

    info = info or get_dist_info()
    seed_everything(train_config.seed)  # same initial weights on every run
    with local_main_first(info):  # one download per machine
        dataset = build_dataset(data_config, train=True)
    class_names = dataset_classes(dataset)
    model_config = dataclasses.replace(
        model_config,
        image_size=data_config.image_size,
        channels=data_config.channels,
        num_classes=num_classes(data_config, dataset),
    )
    device = resolve_device(train_config.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True  # free speed-up on Ampere+, harmless elsewhere
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # fixed input sizes: let cuDNN pick fast kernels
    amp_dtype = resolve_precision(train_config.precision, device)
    task = build_task(model_config, train_config, device, amp_dtype)
    loader = build_loader(dataset, train_config, device)

    sizes = ", ".join(
        f"{name} {count_parameters(net) / 1e6:.2f}M"
        for name, net in task.networks.items()
        if name not in ("G_ema", "ema")
    )
    logger.info(
        "data: %s, %d images, %d classes%s | model: %s",
        data_config.dataset, len(dataset), len(class_names),
        " (conditional)" if model_config.num_classes else "", sizes,
    )  # fmt: skip

    fid_fn = None
    if train_config.fid_every > 0:
        fid_fn = make_fid_fn(data_config, train_config, device)
    state = train(
        task, loader, data_config, model_config, train_config, class_names,
        callback=callback, fid_fn=fid_fn,
    )  # fmt: skip
    return task, state


def make_fid_fn(
    data_config: DataConfig, train_config: TrainConfig, device: torch.device
) -> Callable[[Task], float]:
    """FID of the task's EMA model against the (cached) real-data statistics."""
    from imagegen.data import build_dataset
    from imagegen.fid import InceptionFeatures, dataset_stats, frechet_distance, generated_stats

    extractor = InceptionFeatures().to(device)
    n = train_config.fid_samples
    cache = Path(train_config.output_dir) / f"fid_stats_real_{n}.pt"
    real = dataset_stats(build_dataset(data_config, train=False), extractor, device, n, cache=cache)

    def fid_fn(task: Task) -> float:
        fake = generated_stats(
            task, extractor, n, seed=train_config.seed,
            steps=train_config.sample_steps, guidance_scale=train_config.guidance_scale,
        )  # fmt: skip
        return frechet_distance(*real, *fake)

    return fid_fn


if __name__ == "__main__":
    main()
