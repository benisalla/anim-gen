"""The two ways of learning to generate images, behind one interface.

    GANTask        generator vs discriminator (arch "dcgan" or "resnet")
    DiffusionTask  U-Net denoiser + Gaussian diffusion (arch "unet")

Both expose the same methods, so train.py, sample.py and fid.py never need
to know which family they are driving:

    task.train_step(batches, step) -> {"metric": tensor}   one optimizer step
    task.noise(n, generator)       -> the random input of the sampler
    task.generate(noise, labels)   -> images in [-1, 1] (always from the EMA weights)
    task.state_dict() / task.load_state_dict(state)
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR

from anim_gen.config import ModelConfig, TrainConfig
from anim_gen.diffaugment import diff_augment
from anim_gen.diffusion import GaussianDiffusion
from anim_gen.ema import EMA
from anim_gen.losses import d_loss, g_loss, gradient_penalty, r1_penalty
from anim_gen.models import build_discriminator, build_generator, build_unet
from anim_gen.runtime import autocast, broadcast_module, sync_gradients

Batches = Iterator[tuple[torch.Tensor, torch.Tensor]]


class Task:
    """Shared plumbing. Subclasses fill `networks` and `optimizers`."""

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig | None,
        device: torch.device,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        self.model_config = model_config
        self.train_config = train_config  # None = inference only (no optimizers)
        self.device = device
        self.amp_dtype = amp_dtype
        self.networks: dict[str, nn.Module] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.schedulers: dict[str, LambdaLR] = {}

    @property
    def conditional(self) -> bool:
        return self.model_config.num_classes > 0

    def _next(self, batches: Batches) -> tuple[torch.Tensor, torch.Tensor | None]:
        images, labels = next(batches)
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True) if self.conditional else None
        return images, labels

    def random_labels(self, n: int, generator: torch.Generator | None = None):
        """Uniform class ids (or None for an unconditional model)."""
        if not self.conditional:
            return None
        return torch.randint(0, self.model_config.num_classes, (n,), generator=generator).to(
            self.device
        )

    def state_dict(self) -> dict:
        return {
            "networks": {k: v.state_dict() for k, v in self.networks.items()},
            "optimizers": {k: v.state_dict() for k, v in self.optimizers.items()},
            "schedulers": {k: v.state_dict() for k, v in self.schedulers.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        """Load what this task has; inference tasks simply skip optimizer states."""
        for name, net in self.networks.items():
            net.load_state_dict(state["networks"][name])
        for name, opt in self.optimizers.items():
            opt.load_state_dict(state["optimizers"][name])
        for name, sched in self.schedulers.items():
            sched.load_state_dict(state["schedulers"][name])


class GANTask(Task):
    """Alternate `d_steps` discriminator updates and one generator update."""

    family = "gan"

    def __init__(self, model_config, train_config, device, amp_dtype=None) -> None:
        super().__init__(model_config, train_config, device, amp_dtype)
        self.G = build_generator(model_config).to(device)
        self.networks["G"] = self.G
        cfg = train_config
        if cfg is None:  # inference: the checkpoint's EMA weights are loaded into G_ema
            self.G_ema = EMA(self.G, 0.0)
            self.networks["G_ema"] = self.G_ema
            return
        self.D = build_discriminator(model_config).to(device)
        broadcast_module(self.G)  # every process starts from rank 0's weights
        broadcast_module(self.D)
        self.G_ema = EMA(self.G, cfg.ema_decay, cfg.ema_start)
        self.networks.update(D=self.D, G_ema=self.G_ema)
        betas = (cfg.beta1, cfg.beta2)
        self.optimizers["G"] = Adam(self.G.parameters(), lr=cfg.lr_g, betas=betas)
        self.optimizers["D"] = Adam(self.D.parameters(), lr=cfg.lr_d, betas=betas)

    def noise(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.randn(n, self.model_config.z_dim, generator=generator).to(self.device)

    def _discriminator_step(self, batches: Batches) -> dict[str, torch.Tensor]:
        cfg = self.train_config
        real, y = self._next(batches)
        with torch.no_grad(), autocast(self.device, self.amp_dtype):
            fake = self.G(self.noise(real.shape[0]), y)
        real_in = diff_augment(real, cfg.diffaug)
        fake_in = diff_augment(fake.float(), cfg.diffaug)
        if cfg.r1_gamma > 0:
            real_in = real_in.detach().requires_grad_(True)
        with autocast(self.device, self.amp_dtype):
            real_logits = self.D(real_in, y).float()
            fake_logits = self.D(fake_in, y).float()
        loss = d_loss(cfg.loss, real_logits, fake_logits)
        metrics = {
            "d_loss": loss.detach(),
            "d_real": real_logits.detach().mean(),  # mean logit on real images
            "d_fake": fake_logits.detach().mean(),  # ... and on generated ones
        }
        if cfg.loss == "wgan-gp":  # in fp32: second-order gradients are precision-hungry
            gp = gradient_penalty(lambda x: self.D(x, y), real_in.detach(), fake_in)
            loss = loss + cfg.gp_weight * gp
            metrics["gp"] = gp.detach()
        if cfg.r1_gamma > 0:
            r1 = r1_penalty(real_logits, real_in)
            loss = loss + cfg.r1_gamma / 2 * r1
            metrics["r1"] = r1.detach()

        opt = self.optimizers["D"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sync_gradients(self.D.parameters())
        opt.step()
        return metrics

    def train_step(self, batches: Batches, step: int) -> dict[str, torch.Tensor]:
        cfg = self.train_config
        for _ in range(cfg.d_steps):
            metrics = self._discriminator_step(batches)

        # Generator: fool the (now frozen) discriminator on fresh noise.
        n = cfg.batch_size
        y = self.random_labels(n)
        self.D.requires_grad_(False)  # no gradients wasted on D's weights
        with autocast(self.device, self.amp_dtype):
            fake = self.G(self.noise(n), y)
            fake_logits = self.D(diff_augment(fake, cfg.diffaug), y).float()
        loss = g_loss(cfg.loss, fake_logits)
        opt = self.optimizers["G"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sync_gradients(self.G.parameters())
        opt.step()
        self.D.requires_grad_(True)
        self.G_ema.update(self.G, step)
        metrics["g_loss"] = loss.detach()
        return metrics

    @torch.no_grad()
    def generate(
        self, noise: torch.Tensor, labels: torch.Tensor | None = None, **_
    ) -> torch.Tensor:
        with autocast(self.device, self.amp_dtype):
            return self.G_ema(noise, labels).float()


class DiffusionTask(Task):
    """Denoising diffusion: one MSE regression step per batch."""

    family = "diffusion"

    def __init__(self, model_config, train_config, device, amp_dtype=None) -> None:
        super().__init__(model_config, train_config, device, amp_dtype)
        self.unet = build_unet(model_config).to(device)
        self.diffusion = GaussianDiffusion(
            model_config.timesteps,
            model_config.noise_schedule,
            model_config.prediction,
            null_label=model_config.num_classes if self.conditional else None,
        ).to(device)
        self.networks["unet"] = self.unet
        cfg = train_config
        if cfg is None:
            self.ema = EMA(self.unet, 0.0)
            self.networks["ema"] = self.ema
            return
        broadcast_module(self.unet)
        self.ema = EMA(self.unet, cfg.ema_decay, cfg.ema_start)
        self.networks["ema"] = self.ema
        self.optimizers["unet"] = AdamW(
            self.unet.parameters(),
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )
        warmup = max(1, cfg.warmup_steps)  # linear warmup, then constant (as in DDPM)
        self.schedulers["unet"] = LambdaLR(
            self.optimizers["unet"], lambda s: min(1.0, (s + 1) / warmup)
        )

    def noise(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        c, s = self.model_config.channels, self.model_config.image_size
        return torch.randn(n, c, s, s, generator=generator).to(self.device)

    def train_step(self, batches: Batches, step: int) -> dict[str, torch.Tensor]:
        cfg = self.train_config
        x0, y = self._next(batches)
        if y is not None and self.model_config.cfg_dropout > 0:
            # Sometimes hide the label, so the same network also learns p(x):
            # that is what classifier-free guidance needs at sampling time.
            drop = torch.rand(y.shape[0], device=y.device) < self.model_config.cfg_dropout
            y = torch.where(drop, torch.full_like(y, self.model_config.num_classes), y)
        with autocast(self.device, self.amp_dtype):
            loss = self.diffusion.training_loss(self.unet, x0, y)
        opt = self.optimizers["unet"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sync_gradients(self.unet.parameters())
        max_norm = cfg.grad_clip if cfg.grad_clip > 0 else float("inf")
        grad_norm = nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm)
        opt.step()
        self.schedulers["unet"].step()
        self.ema.update(self.unet, step)
        return {"loss": loss.detach(), "grad_norm": grad_norm.detach()}

    @torch.no_grad()
    def generate(
        self,
        noise: torch.Tensor,
        labels: torch.Tensor | None = None,
        steps: int = 100,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        with autocast(self.device, self.amp_dtype):
            return self.diffusion.sample(
                self.ema, noise, labels, steps=steps, eta=eta,
                guidance_scale=guidance_scale, generator=generator,
            )  # fmt: skip


def build_task(
    model_config: ModelConfig,
    train_config: TrainConfig | None,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
) -> Task:
    """GANTask or DiffusionTask, from the architecture name."""
    cls = DiffusionTask if model_config.family == "diffusion" else GANTask
    return cls(model_config, train_config, device, amp_dtype)


__all__ = ["DiffusionTask", "GANTask", "Task", "build_task"]
