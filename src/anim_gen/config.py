"""All hyperparameters live here, grouped by what they control.

Configs are frozen dataclasses: they cannot be changed by accident after
creation, and they are saved inside every checkpoint so a run can be
reproduced and a checkpoint can rebuild its own model.

PRESETS at the bottom are named starting points (e.g. "cifar10-resnet").
On the command line every field can still be overridden one by one:
    anim-train --preset cifar10-resnet --batch-size 32 --steps 2000
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

DATASETS = ("mnist", "fashion-mnist", "cifar10", "folder", "shapes")
GRAYSCALE_DATASETS = ("mnist", "fashion-mnist")
ARCHS = ("dcgan", "resnet", "unet")  # dcgan / resnet are GANs, unet is a diffusion model
GAN_LOSSES = ("hinge", "nonsaturating", "wgan-gp", "lsgan")
PRECISIONS = ("auto", "fp32", "bf16")


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Which images we learn from and how they become tensors in [-1, 1]."""

    dataset: str = "cifar10"
    data_dir: Path = Path("data")  # download target, or the image folder for "folder"
    image_size: int = 32
    hflip: bool = True  # random horizontal flips (never for digits: a flipped 3 is not a 3)
    conditional: bool = False  # learn p(image | class) from the dataset labels
    limit: int | None = None  # use only the first N images (tiny overfit runs)
    download: bool = True  # torchvision datasets only

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}")
        if self.image_size < 8 or self.image_size & (self.image_size - 1):
            raise ValueError("image_size must be a power of two >= 8 (8, 16, 32, 64, ...)")

    @property
    def channels(self) -> int:
        """1 for grayscale datasets, 3 (RGB) for everything else."""
        return 1 if self.dataset in GRAYSCALE_DATASETS else 3


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Shape of the networks. image_size / channels / num_classes come from the data."""

    arch: str = "resnet"
    image_size: int = 32
    channels: int = 3
    num_classes: int = 0  # 0 = unconditional

    # --- GANs (dcgan, resnet) ---
    z_dim: int = 128  # size of the random noise vector the generator starts from
    g_channels: int = 64  # generator width at full resolution; doubles at each lower one
    d_channels: int = 64  # discriminator width at full resolution
    max_channels: int = 512  # cap on the doubling
    attn_resolution: int = 0  # self-attention block at this resolution (0 = none)
    spectral_norm: bool = True  # spectral norm on every discriminator layer

    # --- diffusion (unet) ---
    unet_channels: int = 64  # width of the first U-Net level
    channel_mult: tuple[int, ...] = (1, 2, 2, 2)  # width multiplier per level
    num_res_blocks: int = 2  # residual blocks per level
    attn_resolutions: tuple[int, ...] = (16,)  # resolutions that get self-attention
    dropout: float = 0.1
    timesteps: int = 1000  # length of the noising process
    noise_schedule: str = "cosine"  # "cosine" | "linear"
    prediction: str = "eps"  # what the U-Net predicts: noise ("eps") or velocity ("v")
    cfg_dropout: float = 0.1  # P(drop the label) in training -> classifier-free guidance

    def __post_init__(self) -> None:
        if self.arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}")
        if self.image_size < 8 or self.image_size & (self.image_size - 1):
            raise ValueError("image_size must be a power of two >= 8")
        if self.noise_schedule not in ("cosine", "linear"):
            raise ValueError("noise_schedule must be cosine or linear")
        if self.prediction not in ("eps", "v"):
            raise ValueError("prediction must be eps or v")
        if not 0.0 <= self.dropout < 1.0 or not 0.0 <= self.cfg_dropout < 1.0:
            raise ValueError("dropout and cfg_dropout must be in [0, 1)")
        if self.image_size >> (len(self.channel_mult) - 1) < 4 and self.arch == "unet":
            raise ValueError("too many U-Net levels for this image_size (lowest must be >= 4)")

    @property
    def family(self) -> str:
        """ "gan" or "diffusion": decides which training task is used."""
        return "diffusion" if self.arch == "unet" else "gan"


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Optimisation and bookkeeping settings for one training run."""

    output_dir: Path = Path("checkpoints/run")
    seed: int = 42
    steps: int = 50_000  # generator (GAN) or denoiser (diffusion) optimizer steps
    batch_size: int = 64  # per process (per GPU)

    # --- GANs ---
    loss: str = "hinge"
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.0  # Adam betas; GANs like a small beta1 (0.0 or 0.5)
    beta2: float = 0.9
    d_steps: int = 1  # discriminator updates per generator update (5 for wgan-gp)
    gp_weight: float = 10.0  # gradient penalty weight (wgan-gp only)
    r1_gamma: float = 0.0  # R1 penalty on real images (0 = off)
    diffaug: str = ""  # DiffAugment policy, e.g. "color,translation,cutout" (small datasets)

    # --- diffusion ---
    lr: float = 2e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    sample_steps: int = 100  # DDIM steps used for preview grids and FID
    guidance_scale: float = 1.0  # classifier-free guidance for previews (1 = off)

    # --- both ---
    ema_decay: float = 0.999  # exponential moving average of the weights (0 = off)
    ema_start: int = 0  # step from which the EMA starts tracking
    precision: str = "auto"  # "auto" (bf16 on Ampere+ GPUs, else fp32) | "fp32" | "bf16"
    num_workers: int = 4
    log_every: int = 100  # steps between log lines
    sample_every: int = 1000  # steps between preview grids (samples/step_XXXXXX.png)
    ckpt_every: int = 5000  # steps between checkpoints
    fid_every: int = 0  # steps between FID evaluations (0 = off; needs Inception weights)
    fid_samples: int = 5000
    num_samples: int = 64  # images in each preview grid
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    resume: Path | None = None  # checkpoint to continue from
    tensorboard: bool = False  # also log to TensorBoard (needs the tensorboard package)
    time_budget_hours: float = 0.0  # save and stop after this long (0 = no limit), e.g. Kaggle

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0 or self.d_steps <= 0:
            raise ValueError("steps, batch_size and d_steps must be positive")
        if self.loss not in GAN_LOSSES:
            raise ValueError(f"loss must be one of {GAN_LOSSES}")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")


def config_to_dict(config: Any) -> dict[str, Any]:
    """Turn a config into plain types so it can be stored in a checkpoint."""
    out = {}
    for key, value in asdict(config).items():
        if isinstance(value, Path):
            value = str(value)
        elif isinstance(value, tuple):
            value = list(value)
        out[key] = value
    return out


def config_from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Inverse of config_to_dict. Unknown keys (from newer versions) are ignored."""
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if isinstance(value, list):
            value = tuple(value)
        elif value is not None and f.name in ("data_dir", "output_dir", "resume"):
            value = Path(value)
        kwargs[f.name] = value
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Presets: flat {field name: value} overrides of the three configs above.
# Field names are unique across DataConfig / ModelConfig / TrainConfig.
# ---------------------------------------------------------------------------

_DCGAN = {
    "arch": "dcgan", "loss": "nonsaturating", "beta1": 0.5, "beta2": 0.999,
    "spectral_norm": False, "z_dim": 100, "ema_decay": 0.999, "ema_start": 1000,
}  # fmt: skip
_RESNET = {
    "arch": "resnet", "loss": "hinge", "beta1": 0.0, "beta2": 0.9, "spectral_norm": True,
    "z_dim": 128, "ema_decay": 0.9999, "ema_start": 1000,
}  # fmt: skip
_UNET = {"arch": "unet", "lr": 2e-4, "ema_decay": 0.9995, "beta1": 0.9, "beta2": 0.999}

PRESETS: dict[str, dict[str, Any]] = {
    # --- MNIST (1 x 32 x 32): minutes on a GPU, sanity checks on a CPU ---
    "mnist-dcgan": {
        **_DCGAN, "dataset": "mnist", "image_size": 32, "hflip": False,
        "g_channels": 32, "d_channels": 32, "max_channels": 256, "steps": 10_000,
    },
    "mnist-diffusion": {
        **_UNET, "dataset": "mnist", "image_size": 32, "hflip": False, "conditional": True,
        "unet_channels": 32, "channel_mult": (1, 2, 2), "attn_resolutions": (8,),
        "steps": 20_000, "batch_size": 128,
    },
    # --- CIFAR-10 (3 x 32 x 32) ---
    "cifar10-dcgan": {
        **_DCGAN, "dataset": "cifar10", "image_size": 32,
        "g_channels": 64, "d_channels": 64, "max_channels": 512, "steps": 50_000,
    },
    "cifar10-resnet": {
        **_RESNET, "dataset": "cifar10", "image_size": 32, "conditional": True,
        "g_channels": 128, "d_channels": 128, "max_channels": 256, "d_steps": 2,
        "attn_resolution": 16, "steps": 60_000, "lr_g": 2e-4, "lr_d": 2e-4,
    },
    "cifar10-diffusion": {
        **_UNET, "dataset": "cifar10", "image_size": 32, "conditional": True,
        "unet_channels": 128, "channel_mult": (1, 2, 2, 2), "attn_resolutions": (16,),
        "dropout": 0.1, "steps": 200_000, "batch_size": 128,
    },
    # --- AFHQ animal faces (cat / dog / wild), prepared with anim-prepare ---
    "afhq64-resnet": {
        **_RESNET, "dataset": "folder", "data_dir": Path("data/afhq64"), "image_size": 64,
        "conditional": True, "g_channels": 64, "d_channels": 64, "max_channels": 512,
        "attn_resolution": 32, "diffaug": "color,translation,cutout", "steps": 100_000,
    },
    "afhq128-resnet": {
        **_RESNET, "dataset": "folder", "data_dir": Path("data/afhq128"), "image_size": 128,
        "conditional": True, "g_channels": 64, "d_channels": 64, "max_channels": 512,
        "attn_resolution": 64, "diffaug": "color,translation,cutout", "r1_gamma": 0.0,
        "steps": 150_000, "batch_size": 32,
    },
    "afhq64-diffusion": {
        **_UNET, "dataset": "folder", "data_dir": Path("data/afhq64"), "image_size": 64,
        "conditional": True, "unet_channels": 128, "channel_mult": (1, 2, 3, 4),
        "attn_resolutions": (16, 8), "prediction": "v", "steps": 300_000, "batch_size": 64,
    },
    "afhq128-diffusion": {
        **_UNET, "dataset": "folder", "data_dir": Path("data/afhq128"), "image_size": 128,
        "conditional": True, "unet_channels": 128, "channel_mult": (1, 1, 2, 3, 4),
        "attn_resolutions": (16, 8), "prediction": "v", "steps": 400_000, "batch_size": 32,
    },
    # --- procedural shapes: no download, used by the tests and quick checks ---
    "shapes-dcgan": {
        **_DCGAN, "dataset": "shapes", "image_size": 16, "g_channels": 16, "d_channels": 16,
        "max_channels": 64, "z_dim": 32, "steps": 500, "num_workers": 0, "ema_start": 100,
    },
    "shapes-diffusion": {
        **_UNET, "dataset": "shapes", "image_size": 16, "unet_channels": 16,
        "channel_mult": (1, 2), "attn_resolutions": (8,), "timesteps": 200,
        "sample_steps": 20, "steps": 500, "num_workers": 0, "warmup_steps": 50,
        "lr": 1e-3, "ema_decay": 0.99,
    },
}  # fmt: skip


def check_presets() -> None:
    """Every preset key must be a field of exactly one config (catches typos)."""
    owners: dict[str, str] = {}
    for cls in (DataConfig, ModelConfig, TrainConfig):
        for f in fields(cls):
            if f.name in owners and cls is not ModelConfig:
                raise AssertionError(f"field {f.name} defined twice")
            owners.setdefault(f.name, cls.__name__)
    for name, preset in PRESETS.items():
        unknown = set(preset) - set(owners)
        if unknown:
            raise AssertionError(f"preset {name}: unknown fields {sorted(unknown)}")


__all__ = [
    "ARCHS",
    "DATASETS",
    "GAN_LOSSES",
    "PRESETS",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "check_presets",
    "config_from_dict",
    "config_to_dict",
]
