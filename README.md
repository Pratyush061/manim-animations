# manim-animations

Simple [Manim](https://www.manim.community/) animations rendered automatically in CI.

## What's here

- `scene.py` — a simple Manim scene (`MyScene`) that writes a title, morphs a circle into a square, then into a formula.
- `.github/workflows/render.yml` — a GitHub Actions workflow that installs Manim, renders the scene, and uploads the video as an artifact.

## How the render works

On every push to `main` (or when triggered manually from the **Actions** tab), the workflow runs
[`MengAiDev/manim_workflow@v1.0.0`](https://github.com/MengAiDev/manim_workflow), which:

1. Installs the system dependencies (Cairo, TeXLive, Pango, ffmpeg, ...).
2. Installs Manim via pip.
3. Renders the scene with `manim scene.py MyScene -qm`.
4. Uploads the resulting `.mp4` as a workflow artifact named `manim-video`.

## Getting the rendered video

1. Open the **Actions** tab and pick the latest **Render Manim Animation** run.
2. Download the `manim-video` artifact at the bottom of the run summary.

## Running locally

```bash
pip install manim
manim -qm scene.py MyScene
```

The video lands in `media/videos/scene/720p30/`.
