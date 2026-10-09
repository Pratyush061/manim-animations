# manim-animations

Simple [Manim](https://www.manim.community/) animations rendered automatically in CI.

## What's here

- `scene.py` — a simple Manim scene (`MyScene`) that writes a title, morphs a circle into a square, then into a formula.
- `.github/workflows/render.yml` — a GitHub Actions workflow that installs Manim, renders the scene, and publishes the video.
- `rendered/MyScene.mp4` — the latest rendered video, committed automatically by the workflow (the easiest place to watch or download it).

## Where to get the video

There are two places, both produced by the same run:

1. **`rendered/MyScene.mp4` in the repository** — open the file and GitHub plays it inline, with a download button. The workflow commits it after every successful render.
2. **The `manim-video` artifact** on a workflow run — open the **Actions** tab, click the latest successful **Render Manim Animation** run, and scroll to the **Artifacts** section at the very bottom of that run's summary page. The artifact is a `.zip` containing the `.mp4`.

Note: artifacts live on the *run summary page* (bottom), not in the Actions list itself. A failed run (for example the very first one) has no artifact.

## How the render works

On every push to `main` (or when triggered manually from the **Actions** tab), the workflow runs
[`MengAiDev/manim_workflow@v1.0.0`](https://github.com/MengAiDev/manim_workflow), which:

1. Installs the system dependencies (Cairo, TeXLive, Pango, ffmpeg, ...).
2. Installs Manim via pip.
3. Renders the scene with `manim scene.py MyScene -qm`.
4. Uploads the resulting `.mp4` as a workflow artifact named `manim-video`.

The workflow then commits the rendered file to `rendered/MyScene.mp4`.

## Running locally (optional)

This is only if you want to render on your own machine — it has nothing to do with the GitHub Actions setup:

```bash
pip install manim
manim -qm scene.py MyScene
```

The video lands in `media/videos/scene/720p30/`.
