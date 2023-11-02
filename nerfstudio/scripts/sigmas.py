"""
sigmas.py - modified from eval.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import tyro

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class RenderSigmas:
    """Load a checkpoint, generate a movie of the sigmas for the evaluation set, and
    store both the xyz_locations and the sigma values to disk for future usage."""

    # Path to config YAML file.
    load_config: Path
    # Save the xyz locations to disk.
    save_xyz: bool = False
    # Load the xyz locations from disk.
    load_xyz: bool = False
    # Path to the xyz locations to load from disk.
    load_xyz_path: Path = Path('None')
    # Frames per second for the movie.
    fps: int = 20

    def main(self) -> None:
        """Main function."""
        # import torch
        # from scratch.algos.grid import sample_grid_box
        # _, pipeline, _, _ = eval_setup(self.load_config)
        # pts = sample_grid_box(torch.tensor([200,200,200]), torch.device('cuda'))
        # sigma = pipeline.model.field.get_density_at_positions(pts.reshape(-1, 1, 3))
        # breakpoint()

        assert self.save_xyz or self.load_xyz, "Must either save or load xyz locations."
        mode: str
        if self.load_xyz:
            assert os.path.exists(self.load_xyz_path), "Path to xyz locations does not exist."
            mode = 'load'
        else:
            mode = 'save'

        _, pipeline, _, _ = eval_setup(self.load_config)
        pipeline.get_eval_image_sigma_viz(mode=mode, load_xyz_path=self.load_xyz_path, fps=self.fps)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderSigmas).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RenderSigmas)  # noqa
