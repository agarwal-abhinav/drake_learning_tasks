# Part of the IROS Public Release (eval repo)

This repo (`drake_learning_tasks`) provides **evaluation for `long_context_planar_pushing`** and
will be **vendored into the public umbrella repo** (`diffusion-policy-experiment`), not published
on its own. It has **its own conda environment** (see this repo's install instructions).

## Master plan / source of truth
The full release plan, decisions, and phase status live in the umbrella repo on its `release/iros`
branch: `diffusion-policy-experiment/RELEASE_PROGRESS.md`. Read that first.

## Constraints (same as umbrella)
- Work only on branch `release/iros`; never touch `main`.
- Public repo is built with CLEAN history (squash) — this repo's history is NOT published.
- Vendored into umbrella under `eval/long_context_pushing/`.

## Status
- `release/iros` branch created as the staging point for release cleanup of the eval code.
- No eval-code pruning done yet — that happens during umbrella Phase 3 (vendoring).

> Delete this file before the final publish squash — it is release scaffolding.
