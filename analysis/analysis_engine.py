from datetime import datetime
from pathlib import Path
from typing import Optional

from analysis.visualization import ensure_dir, save_heatmap, save_patch_grid, save_sorted_curve


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

    def save_step_state(self, model, step: int) -> None:
        hebb = model.discrimination_layer.organizer.potential_hebb
        save_patch_grid(
            hebb,
            self.hebb_dir / f"hebb_step_{step:04d}_grid.png",
            image_hw=self.image_hw,
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
            per_patch_minmax=True,
        )

    def save_organize_state(self, model, step: int) -> None:
        dl = model.discrimination_layer
        save_patch_grid(
            dl.neuron_weights.detach(),
            self.weights_dir / f"weights_step_{step:04d}_grid.png",
            image_hw=self.image_hw,
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
            per_patch_minmax=True,
        )
        save_sorted_curve(
            dl.organizer.lr_vec.detach(),
            self.lr_dir / f"lr_step_{step:04d}_sorted.png",
            title=f"Sorted lr_vec at step {step}",
            ylabel="lr",
        )
        save_heatmap(
            dl.neuron_correlation_matrix.detach(),
            self.corr_dir / f"corr_step_{step:04d}_heatmap.png",
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
