# Legacy notebook (reference only)

`animal_image_generator_gans.ipynb` is the first, experimental version of the
project: a Colab notebook with a ResBlock-based GAN trained on a zip of animal
photos from Google Drive. It is **not maintained** and will not run outside
that Colab/Drive setup.

It is kept for history. Everything it did is in `src/imagegen/`, and the
issues below are fixed there:

| In the notebook | Effect | In `src/imagegen/` |
|---|---|---|
| The generator trains on `fixed_noise`, the same 32 vectors at every step | the generator only ever learns 32 images, so it memorises instead of generalising | fresh noise every step (`tasks.py`); fixed noise is used only for preview grids |
| `get_noise()` calls `set_seed()` | every call returns the *same* noise | one seeded generator for previews only |
| The D step uses `fake_images` without `detach()`, then `backward(retain_graph=True)` | gradients flow into the generator during the D step; extra memory | fakes are generated under `no_grad` for the D step; D is frozen during the G step |
| `Sigmoid` + `BCELoss` | numerically unstable once D is confident | raw logits + `softplus` (= BCE with logits), plus hinge / WGAN-GP / LSGAN |
| `InstanceNorm2d(track_running_stats=True)` in G | train and eval behave differently | BatchNorm / conditional BatchNorm, EMA generator for sampling |
| Train and val transforms use different normalisations | real images seen with two different colour scales | one transform: [-1, 1] everywhere |
| Checkpoint every 50 batches to Drive, only the weights | slow; a restart loses optimizer state and the step count | atomic `last.pt` with optimizers, EMA, RNG and configs; `--resume` |
| One fixed architecture, 128×128 only | | DCGAN, ResNet-SNGAN, diffusion U-Net; any power-of-two size |

Use `src/imagegen/` for everything else (see the main README).
