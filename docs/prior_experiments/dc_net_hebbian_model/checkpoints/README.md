# checkpoints/

Named, reusable layer weights - distinct from `RESULT/<experiment>/<timestamp>/`,
which is a per-run audit trail (best/final .pth for reproducing one specific result,
gitignored via the root `.gitignore`'s `*.pth` rule since those can be large).

A file here is a layer someone deliberately promoted for reuse in a *different*
experiment later (e.g. pulling a trained discrimination layer into a parallel-branch
or fuse-expand architecture). Saved and loaded via `core/checkpointing.py`
(`save_layer` / `load_layer`), never by hand - that keeps `manifest.json` (the index
of what's here, with dims and provenance) in sync with the actual `.pt` files.

Unlike `RESULT/.../*.pth`, files here ARE tracked in git on purpose - the point of a
named checkpoint is that it's actually present when the repo is cloned, not something
you have to separately regenerate or fetch. If these grow large or numerous, switch to
Git LFS for this folder rather than gitignoring it (gitignoring would silently break
`load_layer()` for anyone without the original file).
