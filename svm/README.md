# Support Vector Machines — Manim explainer

A ~12-minute, 3Blue1Brown-grade explainer on **Support Vector Machines**, built
scene by scene as six parts (`part-01` … `part-06`, scenes `S01` … `S15`).
It runs the full arc: the widest-street intuition, the support vectors, the
optimisation problem, the soft margin and `C`, the wall of non-separable data,
the lift into 3-D, the kernel trick, and the kernel menu — closing with a
callback to the opening scatter.

On-screen text is English; the accompanying narration is Hinglish (see the
voiceover script in the topic notes).

## Layout

| File | Scenes | What it covers |
|------|--------|----------------|
| `svm_common.py` | — | Palette, datasets, geometry helpers, base scene |
| `part-01-hook.py` | `S01_Hook`, `S02_ManyLines` | The promise; too many valid lines |
| `part-02-margin.py` | `S03_NarrowVsWide`, `S04_TheStreet`, `S05_SupportVectors` | The margin, side by side; only SVs matter |
| `part-03-math.py` | `S06_Hyperplane`, `S07_Margin`, `S08_Optimization` | The equations, term by term |
| `part-04-soft-margin.py` | `S09_WhyWideMargin`, `S10_SoftMargin` | Robustness; slack variables and `C` |
| `part-05-kernels.py` | `S11_TheWall`, `S12_TheLift`, `S13_KernelTrick`, `S14_KernelMenu` | Non-separable data → the kernel trick |
| `part-06-recap.py` | `S15_Recap` | The three takeaways |

## Render

```bash
pip install manim            # needs Python >= 3.11
manim -qm part-02-margin.py S03_NarrowVsWide        # one scene, preview
manim -qh part-05-kernels.py S12_TheLift            # 1080p
```

Every scene in order (720p30), then stitched into one video:

```bash
for spec in \
  "part-01-hook.py:S01_Hook" "part-01-hook.py:S02_ManyLines" \
  "part-02-margin.py:S03_NarrowVsWide" "part-02-margin.py:S04_TheStreet" \
  "part-02-margin.py:S05_SupportVectors" \
  "part-03-math.py:S06_Hyperplane" "part-03-math.py:S07_Margin" \
  "part-03-math.py:S08_Optimization" \
  "part-04-soft-margin.py:S09_WhyWideMargin" "part-04-soft-margin.py:S10_SoftMargin" \
  "part-05-kernels.py:S11_TheWall" "part-05-kernels.py:S12_TheLift" \
  "part-05-kernels.py:S13_KernelTrick" "part-05-kernels.py:S14_KernelMenu" \
  "part-06-recap.py:S15_Recap" ; do
  f="${spec%%:*}"; s="${spec##*:}"
  manim -qm --disable_caching --media_dir media "$f" "$s"
done
```

Rendered automatically by [`.github/workflows/render-svm.yml`](../.github/workflows/render-svm.yml),
which concatenates the scenes and commits the result to
[`rendered/svm_explainer.mp4`](../rendered/svm_explainer.mp4).

## Requirements

- Manim Community edition (verified against **0.22.0**).
- A LaTeX distribution (TeX Live) plus **dvisvgm** for `MathTex` (used
  throughout Parts 3–5). On Debian/Ubuntu the CI installs:
  `texlive-latex-base texlive-latex-recommended texlive-latex-extra
  texlive-fonts-recommended texlive-science dvisvgm`.
- No external image, SVG or audio assets — every visual is drawn with Manim
  primitives, and fonts use Manim's default.

## Design

- **Palette** (fixed once in `svm_common.py`, used everywhere): class −1
  `#58C4DD`, class +1 `#FFD166`, margin/street `#83C167`, violations `#FC6255`,
  ink `#ECECEC`, muted `#8A93A6`, on a near-black `#0A0A0A` background.
- **Correct by construction.** `separator()` returns the maximum-margin
  hyperplane analytically from two support vectors, so the rails really do
  touch the points the animation claims they touch — never eyeballed. The
  side-by-side comparisons reuse the same orientation and only change the
  margin width, so the contrast is honest.
- **One idea per beat.** Every reveal is held with a `self.wait()` before the
  next begins, and the three side-by-side comparisons (narrow vs wide,
  all-points vs support-vectors, the three kernels) are laid out as literal
  two- and three-panel frames.
- **Assets:** none.

## Verification status

The six parts were authored and **statically verified** against the Manim
0.22 source — every class, method and keyword argument used was checked
against that source, all files compile, the on-screen text fits the frame, and
the separator/margin geometry is asserted numerically. The authoring sandbox
had no Linux `manimpango`/`pycairo` wheel and no compiler, so the scenes could
not be frame-rendered there; the CI workflow in this repo performs the render.
