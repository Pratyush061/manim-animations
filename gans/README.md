# GANs Explainer — Manim animation

A ~18-minute, 3Blue1Brown-grade explainer on **Generative Adversarial Networks**,
written scene-by-scene as 17 Manim scenes (`S00` … `S16`) in
[`gans_explainer.py`](gans_explainer.py).

The film follows the classic arc: the forger-vs-detective metaphor, the two
networks, the latent-space warping map, one full training round, the minimax
objective, the optimal discriminator, convergence to Jensen–Shannon divergence,
then the two failure modes (vanishing gradients, mode collapse), the fixes
(non-saturating loss, Wasserstein GAN, unrolled GANs), the GAN zoo, evaluation,
and the modern diffusion landscape.

## Rendered output

Rendered automatically by [`.github/workflows/render-gans.yml`](../.github/workflows/render-gans.yml),
which renders every scene at 720p30, concatenates them in order, and commits the
result to [`rendered/gans_explainer.mp4`](../rendered/gans_explainer.mp4).

## Render locally

```bash
pip install manim            # needs Python >= 3.11
manim -qm gans_explainer.py S00_ColdOpen      # one scene (preview)
```

Render everything and stitch it together:

```bash
for S in S00_ColdOpen S01_GenerationProblem S02_AveragingTrap \
         S03_CounterfeiterAndPolice S04_TwoNetworks S05_WarpingMap \
         S06_TrainingRound S07_Minimax S08_OptimalDiscriminator \
         S09_Convergence S10_NoMonotoneLoss S11_VanishingGradients \
         S12_ModeCollapse S13_Fixes S14_GANZoo \
         S15_EvaluationAndLandscape S16_Closing; do
  manim -qm --disable_caching gans_explainer.py "$S"
done
```

## Requirements

- Manim Community edition (verified against **0.22.0**).
- A LaTeX distribution (TeX Live) plus **dvisvgm** for `MathTex` (scenes S07–S09).
  On Debian/Ubuntu the CI installs: `texlive-latex-base texlive-latex-recommended
  texlive-latex-extra texlive-fonts-recommended texlive-science dvisvgm`.
- No external image, SVG or audio assets — every visual is drawn with Manim
  primitives, and fonts use Manim's default.

## Design

- **Palette** (fixed once, used everywhere): real data `#58C4DD`, generator
  fakes `#FFD166`, discriminator `#FC6255`, equilibrium `#83C167`, ink
  `#ECECEC`, muted `#8A93A6`, on a near-black `#0E1116` background.
- **Motion carries meaning**: the distribution curves in S09/S11 are driven by a
  `ValueTracker` + `always_redraw`, so every frame is correct-by-construction
  rather than keyframed.
- Scenes are short and independent for fast iteration; a shared `GANScene` base
  sets the background and exposes a zoomable camera.
