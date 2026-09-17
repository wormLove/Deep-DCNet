from datetime import datetime
from pathlib import Path
from typing import Optional

from analysis.visualization import ensure_dir, save_heatmap, save_patch_grid, save_sorted_curve


def _near_square_hw(n: int):
    """(h, w) with h * w == n and h as close to sqrt(n) as possible."""
    h = int(n ** 0.5)
    while h > 1 and n % h != 0:
        h -= 1
    return (h, n // h)


class AnalysisEngine:
    """
    Save visual diagnostics for discrimination-layer learning.
    """

    def __init__(
        self,
        base_dir: Path,
        run_name: str = "gpu_rebuild_discrimination",
        image_hw=(28, 28),
        grid_rows: int = 25,
        grid_cols: int = 40,
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = ensure_dir(Path(base_dir) / run_name / timestamp)
        self.weights_dir = ensure_dir(self.run_dir / "weights")
        self.hebb_dir = ensure_dir(self.run_dir / "potential_hebb")
        self.lr_dir = ensure_dir(self.run_dir / "lr_vec")
        self.corr_dir = ensure_dir(self.run_dir / "correlation")
        self.ckpt_dir = ensure_dir(self.run_dir / "checkpoints")
        self.image_hw = image_hw
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)

    def _layers(self, model):
        """
        Yields (file_tag, discrimination_layer, image_hw) for every
        discrimination layer in the model.

        Single-layer models (model.discrimination_layer) yield one entry
        with an empty tag, so their output filenames are unchanged.
        Stacked models (model.discrimination_layers) yield one entry per
        layer, tagged "_L0", "_L1", ... Only layer 0's columns are input
        images; a deeper layer's columns live in the previous layer's
        activation space, so they are drawn on a near-square grid purely as
        a visual fingerprint, not as something image-like.
        """
        if hasattr(model, "discrimination_layers"):
            for i, dl in enumerate(model.discrimination_layers):
                hw = self.image_hw if i == 0 else _near_square_hw(dl.in_dim)
                yield f"_L{i}", dl, hw
        else:
            yield "", model.discrimination_layer, self.image_hw

    def save_step_state(self, model, step: int) -> None:
        for tag, dl, hw in self._layers(model):
            save_patch_grid(
                dl.organizer.potential_hebb,
                self.hebb_dir / f"hebb_step_{step:04d}{tag}_grid.png",
                image_hw=hw,
                grid_rows=self.grid_rows,
                grid_cols=self.grid_cols,
                per_patch_minmax=True,
            )

    def save_organize_state(self, model, step: int) -> None:
        for tag, dl, hw in self._layers(model):
            save_patch_grid(
                dl.neuron_weights.detach(),
                self.weights_dir / f"weights_step_{step:04d}{tag}_grid.png",
                image_hw=hw,
                grid_rows=self.grid_rows,
                grid_cols=self.grid_cols,
                per_patch_minmax=True,
            )
            save_sorted_curve(
                dl.organizer.lr_vec.detach(),
                self.lr_dir / f"lr_step_{step:04d}{tag}_sorted.png",
                title=f"Sorted lr_vec at step {step}{tag}",
                ylabel="lr",
            )
            save_heatmap(
                dl.neuron_correlation_matrix.detach(),
                self.corr_dir / f"corr_step_{step:04d}{tag}_heatmap.png",
                cmap="viridis",
                percentile_clip=99.0,
            )

    def save_checkpoint(self, model, step: int) -> Path:
        path = self.ckpt_dir / f"model_step_{step:04d}.pth"
        model.save_model(str(path))
        return path

    def save_final_checkpoint(self, model) -> Path:
        path = self.ckpt_dir / "final.pth"
        model.save_model(str(path))
        return path
