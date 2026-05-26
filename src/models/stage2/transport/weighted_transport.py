"""
Weighted Transport with Flow Matching support
Adds spatial weighting and latent consistency loss
Created: 2025-11-05
"""

import torch as th
from .transport import Transport
from .utils import mean_flat
from . import path


class WeightedTransport(Transport):
    """
    Transport with spatial weighting and Flow Matching support

    Features:
    1. Spatial weighting for sparse BEV data (背景vs有效区域)
    2. Flow Matching: start from noisy data instead of random noise
    3. Latent consistency loss: regularize latent space
    """

    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        time_dist_type,
        time_dist_shift,
        train_eps,
        sample_eps,
        # Flow Matching parameters
        use_flow_matching=False,
        noise_scale=0.1,
        latent_loss_weight=0.2,
        # Spatial weighting parameters
        bg_weight=0.1,
        bg_threshold=10.0,
    ):
        """
        Initialize WeightedTransport

        Args:
            use_flow_matching: whether to use flow matching (start from noisy)
            noise_scale: noise level added to noisy starting point
            latent_loss_weight: weight for latent consistency loss
            bg_weight: weight for background pixels (0.1 = 10% weight)
            bg_threshold: threshold to determine background pixels
        """
        super().__init__(
            model_type=model_type,
            path_type=path_type,
            loss_type=loss_type,
            time_dist_type=time_dist_type,
            time_dist_shift=time_dist_shift,
            train_eps=train_eps,
            sample_eps=sample_eps,
        )

        # Flow Matching config
        self.use_flow_matching = use_flow_matching
        self.noise_scale = noise_scale
        self.latent_loss_weight = latent_loss_weight

        # Spatial weighting config
        self.bg_weight = bg_weight
        self.bg_threshold = bg_threshold

    def get_loss_info(self):
        """Return loss configuration for saving"""
        return {
            'use_flow_matching': self.use_flow_matching,
            'noise_scale': self.noise_scale,
            'latent_loss_weight': self.latent_loss_weight,
            'bg_weight': self.bg_weight,
            'bg_threshold': self.bg_threshold,
        }

    def compute_spatial_weight(self, x_clean):
        """
        Compute spatial weighting mask

        Args:
            x_clean: clean BEV latent [B, C, H, W]

        Returns:
            weight_mask: [B, 1, H, W] spatial weights
        """
        # Background: pixels with low intensity across channels
        # Shape: [B, 1, H, W]
        pixel_intensity = th.mean(th.abs(x_clean), dim=1, keepdim=True)

        # Background mask: True where intensity < threshold
        bg_mask = pixel_intensity < self.bg_threshold

        # Weight mask: bg_weight for background, 1.0 for foreground
        weight_mask = th.where(bg_mask, self.bg_weight, 1.0)

        return weight_mask

    def training_losses(
        self,
        model,
        x1,
        model_kwargs=None,
        x_noisy=None,
    ):
        """
        Compute training losses with Flow Matching and spatial weighting

        Args:
            model: backbone model (velocity predictor)
            x1: clean data [B, C, H, W]
            model_kwargs: additional model arguments (e.g., condition)
            x_noisy: noisy data for flow matching [B, C, H, W]

        Returns:
            terms: dict with 'loss', 'velocity_loss', 'latent_loss', 'pred'
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Sample time and starting point
        if self.use_flow_matching and x_noisy is not None:
            # Flow Matching: start from noisy + small noise
            t, x0, x1 = self.sample(x1, x_noisy=x_noisy, noise_scale=self.noise_scale)
        else:
            # Standard: start from random noise
            t, x0, x1 = self.sample(x1)

        # Plan the path: get xt and target velocity ut
        t, xt, ut = self.path_sampler.plan(t, x0, x1)

        # Model prediction
        model_output = model(xt, t, **model_kwargs)
        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C), \
            f"Model output shape {model_output.shape} doesn't match expected {xt.shape}"

        terms = {}
        terms['pred'] = model_output

        # 1. Velocity loss (with spatial weighting)
        velocity_error = (model_output - ut) ** 2  # [B, C, H, W]

        # Compute spatial weight
        weight_mask = self.compute_spatial_weight(x1)  # [B, 1, H, W]

        # Weighted velocity loss
        weighted_velocity_loss = mean_flat(weight_mask * velocity_error)
        terms['velocity_loss'] = weighted_velocity_loss.mean()  # 转为标量

        # 2. Latent consistency loss (Flow Matching specific)
        if self.use_flow_matching and x_noisy is not None:
            # Regularize: latent of noisy should be close to latent of clean
            # This encourages the model to learn consistent latent representations
            latent_diff = (x0 - x1) ** 2  # Difference between noisy and clean latents
            latent_loss = mean_flat(latent_diff)
            terms['latent_loss'] = latent_loss.mean()  # 转为标量

            # Combined loss
            velocity_weight = 1.0 - self.latent_loss_weight
            terms['loss'] = (velocity_weight * weighted_velocity_loss.mean() +
                            self.latent_loss_weight * latent_loss.mean())
        else:
            # No latent loss without flow matching
            terms['latent_loss'] = th.tensor(0.0, device=x1.device)
            terms['loss'] = weighted_velocity_loss.mean()  # 转为标量

        return terms


def create_transport(
    *,
    model_type="velocity",
    path_type="linear",
    loss_type="none",
    time_dist_type="uniform",
    time_dist_shift=1.0,
    train_eps=1e-3,
    sample_eps=1e-3,
    use_flow_matching=False,
    noise_scale=0.1,
    latent_loss_weight=0.2,
    bg_weight=0.1,
    bg_threshold=10.0,
):
    """
    Factory function to create WeightedTransport

    Converts string arguments to enums for convenience
    """
    from .transport import ModelType, PathType, WeightType

    # Convert strings to enums
    model_type_map = {
        "noise": ModelType.NOISE,
        "score": ModelType.SCORE,
        "velocity": ModelType.VELOCITY,
    }
    path_type_map = {
        "linear": PathType.LINEAR,
        "gvp": PathType.GVP,
        "vp": PathType.VP,
    }
    loss_type_map = {
        "none": WeightType.NONE,
        "velocity": WeightType.VELOCITY,
        "likelihood": WeightType.LIKELIHOOD,
    }

    model_type_enum = model_type_map.get(model_type.lower(), ModelType.VELOCITY)
    path_type_enum = path_type_map.get(path_type.lower(), PathType.LINEAR)
    loss_type_enum = loss_type_map.get(loss_type.lower(), WeightType.NONE)

    return WeightedTransport(
        model_type=model_type_enum,
        path_type=path_type_enum,
        loss_type=loss_type_enum,
        time_dist_type=time_dist_type,
        time_dist_shift=time_dist_shift,
        train_eps=train_eps,
        sample_eps=sample_eps,
        use_flow_matching=use_flow_matching,
        noise_scale=noise_scale,
        latent_loss_weight=latent_loss_weight,
        bg_weight=bg_weight,
        bg_threshold=bg_threshold,
    )
