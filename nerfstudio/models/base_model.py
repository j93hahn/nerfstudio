# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Base Model implementation which takes in RayBundles
"""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes)
from nerfstudio.model_components.scene_colliders import NearFarCollider


# Model related configs
@dataclass
class ModelConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: Model)
    """target class to instantiate"""
    enable_collider: bool = True
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = to_immutable_dict({"near_plane": 2.0, "far_plane": 6.0})
    """parameters to instantiate scene collider with"""
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0})
    """parameters to instantiate density field with"""
    eval_num_rays_per_chunk: int = 4096
    """specifies number of rays per chunk during eval"""
    prompt: Optional[str] = None
    """A prompt to be used in text to NeRF models"""


class Model(nn.Module):
    """Model class
    Where everything (Fields, Optimizers, Samplers, Visualization, etc) is linked together. This should be
    subclassed for custom NeRF model.

    Args:
        config: configuration for instantiating model
        scene_box: dataset scene box
    """

    config: ModelConfig

    def __init__(
        self,
        config: ModelConfig,
        scene_box: SceneBox,
        num_train_data: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.scene_box = scene_box
        self.render_aabb: Optional[SceneBox] = None  # the box that we want to render - should be a subset of scene_box
        self.num_train_data = num_train_data
        self.kwargs = kwargs
        self.collider = None

        self.populate_modules()  # populate the modules
        self.callbacks = None
        # to keep track of which device the nn.Module is on
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks that run functions at the specified training iterations."""
        return []

    def populate_modules(self):
        """Set the necessary modules to get the network working."""
        # default instantiates optional modules that are common among many networks
        # NOTE: call `super().populate_modules()` in subclasses

        if self.config.enable_collider:
            assert self.config.collider_params is not None
            self.collider = NearFarCollider(
                near_plane=self.config.collider_params["near_plane"], far_plane=self.config.collider_params["far_plane"]
            )

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """

    @abstractmethod
    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """

    @abstractmethod
    def get_sigmas(self, ray_bundle: RayBundle) -> Dict[str, Union[torch.Tensor, List]]:
        """Obtain the sigmas for the ray bundle

        Modeled after the get_outputs() method.

        Args:
            ray_bundle: ray bundle to compute sigmas for

        Returns:
            sigmas for the ray bundle
        """

    def forward(self, ray_bundle: RayBundle) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle)

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """

        return {}

    @abstractmethod
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                if not torch.is_tensor(output):
                    # TODO: handle lists of tensors as well
                    continue
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

    @torch.no_grad()
    def get_sigmas_from_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the sigmas at a given percentile range of the model's
        outputted weights from the volumetric rendering equation along each ray.

        Modeled after the get_outputs_for_camera_ray_bundle() function.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            if self.collider is not None:
                ray_bundle = self.collider(ray_bundle)
            outputs = self.get_sigmas(ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = np.concatenate(outputs_list).reshape(image_height, image_width, -1)  # type: ignore
        return outputs

    def viz_histograms(
        self,
        weights: np.ndarray,
        sigmas: np.ndarray,
        xyz_samples: np.ndarray,
        percentile: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the weights histogram of multiple rays simultaneously

        Args:
            weights: N_rays x N_samples
            sigmas: N_rays x N_samples
            xyz_samples: N_rays x N_samples x 3
            percentile: percentile to compute the value at

        Returns:
            indices of weights histogram
            weights at indices
            sigmas at indices
            xyz locations at indices
        """
        _idxs = np.zeros(weights.shape[0], dtype=np.int32)
        _weights = np.zeros(weights.shape[0], dtype=np.float32)
        _sigmas = np.zeros(weights.shape[0], dtype=np.float32)
        _xyz_locs = np.zeros((weights.shape[0], 3), dtype=np.float32)

        # mask will store the indices of the rays that have not yet been processed
        # as false; once a ray has been processed, its index will be set to true
        mask = np.zeros(weights.shape[0], dtype=bool)
        _weights_sum = weights.sum(axis=-1)

        # if the sum of the weights is 0, the ray passed through empty space; apply a
        # mask to that ray as it will not contribute to the final image
        if np.any(_weights_sum == 0):
            mask[_weights_sum == 0] = True
            if mask.sum() == weights.shape[0]:
                return _idxs, _weights, _sigmas, _xyz_locs

        # return the maximum weight if it is >= 50% of the total sum of the weights; this
        # value must divide the weights histogram into two equal parts
        np.seterr(divide='ignore', invalid='ignore')    # ignore divide by zero warnings
        if np.any((weights[~mask].max(axis=-1) / _weights_sum[~mask]) >= 0.5):
            _wmax = np.nan_to_num(weights.max(axis=-1) / _weights_sum, nan=0.0)
            _wmax = (_wmax >= 0.5) & (~mask)
            _idxs[_wmax] = weights.argmax(axis=-1)[_wmax]
            _weights[_wmax] = weights[_wmax, _idxs[_wmax]]
            _sigmas[_wmax] = sigmas[_wmax, _idxs[_wmax]]
            _xyz_locs[_wmax] = xyz_samples[_wmax, _idxs[_wmax]]

            # apply a mask to the rays that have been processed
            mask[_wmax] = True
            if mask.sum() == weights.shape[0]:
                return _idxs, _weights, _sigmas, _xyz_locs

        # normalize the weights of each ray to sum 1 and compute its cumulative distribution function
        weights_cum = np.nan_to_num(weights / _weights_sum[..., None], nan=0.0)
        weights_cum = np.cumsum(weights_cum, axis=-1)

        # extract the first weight value at the specified percentile of the CDF for each ray
        weights_cum = (weights_cum >= percentile)
        _wpercentile = np.argmax(weights_cum, axis=-1) # argmax returns the first index where the condition is met

        # store the index, weight, and xyz location of the given ray
        _idxs[~mask] = _wpercentile[~mask]
        _weights[~mask] = weights[~mask, _idxs[~mask]]
        _sigmas[~mask] = sigmas[~mask, _idxs[~mask]]
        _xyz_locs[~mask] = xyz_samples[~mask, _idxs[~mask]]

        # turn divide by zero warnings back on
        np.seterr(divide='warn', invalid='warn')

        # return the indices, weights, and xyz locations of the rays
        return _idxs, _weights, _sigmas, _xyz_locs

    def create_single_sigma_viz(self, _sigmas, pose, height, width):
        import matplotlib.pyplot as plt
        from matplotlib.colors import AsinhNorm, LogNorm, Normalize
        from mpl_toolkits.axes_grid1 import ImageGrid

        fig = plt.figure(figsize=(5, 6))
        grid = ImageGrid(
            fig, 111, nrows_ncols=(1, 1),
            cbar_location="right", cbar_mode="edge", cbar_size="7%", cbar_pad=0.15,
        )

        lower = 1e-2
        upper = 1e4

        _sigmas[_sigmas < lower] = lower
        _sigmas[_sigmas > upper] = upper

        h = grid[0].imshow(_sigmas.reshape(height, width), cmap='viridis', norm=LogNorm(lower, upper))
        grid[0].set_title(f'Sigma Visualizations at Pose {pose}')
        grid[0].get_xaxis().set_visible(False)
        grid[0].get_yaxis().set_visible(False)

        plt.colorbar(h, cax=grid.cbar_axes[0])
        plt.savefig(f'sigmas_pose{pose}.png', dpi=300)
        plt.close()

    @abstractmethod
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.
        TODO: This shouldn't return a loss

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """

    def load_model(self, loaded_state: Dict[str, Any]) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: dictionary of pre-trained model states
        """
        state = {key.replace("module.", ""): value for key, value in loaded_state["model"].items()}
        self.load_state_dict(state)  # type: ignore

    def update_to_step(self, step: int) -> None:
        """Called when loading a model from a checkpoint. Sets any model parameters that change over
        training to the correct value, based on the training step of the checkpoint.

        Args:
            step: training step of the loaded checkpoint
        """
