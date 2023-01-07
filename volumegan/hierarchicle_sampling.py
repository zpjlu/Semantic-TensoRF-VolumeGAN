# python3.7
"""Contains the function to sample the points in 3D space."""

from argparse import RawDescriptionHelpFormatter
import torch
import torch.nn.functional as F
from .renderer import renderer

__all__ = ['HierarchicalSampling']
_CLAMP_MODE = ['softplus', 'relu', 'mipnerf']
class HierarchicalSampling(object):
    """Hierarchically samples the points according to the coarse results.

    Args:
        last_back:
        white_back:
        clamp_mode:
        render_mode:
        fill_mode:
        max_depth:
        num_per_group:
    
    """

    def __init__(self,
                 last_back=False,
                 white_back=False,
                 clamp_mode=None,
                 render_mode=None,
                 fill_mode=None,
                 max_depth=None,
                 num_per_group=None):
        """Initializes with basic settings."""

        self.last_back = last_back
        self.white_back = white_back
        self.clamp_mode = clamp_mode
        self.render_mode = render_mode
        self.fill_mode = fill_mode
        self.max_depth = max_depth
        self.num_per_group = num_per_group

    def __call__(self, coarse_rgbs, coarse_sigmas, pts_z, ray_origins, ray_dirs, noise_std=0.5, sample_model=1, **kwargs):
        """
        Args:
            coarse_rgbs: (batch_size, num_of_rays, num_steps, 3) or (batch_size, H, W, num_steps, 3)
            coarse_sigmas: (batch_size, num_of_rays, num_steps, 1) or (batch_size, H, W, num_steps, 1)
            pts_z: (batch_size, num_of_rays, num_steps, 1) or (batch_size, H, W, num_steps, 1)
            ray_origins: (batch_size, num_of_rays, 3) or (batch_size, H, W, 3)
            ray_dirs: Ray directions. (batch_size, num_of_rays, 3) or (batch_size, H, W, 3)
            noise_std: 

        Returns: dict()
            pts: (batch_size, num_of_rays, num_steps, 3) or (batch_size, H, W, num_steps, 3)
            pts_z: (batch_size, num_of_rays, num_steps, 1) or (batch_size, H, W, num_steps, 1)

        """

        for key, value in kwargs.items():
            if key in self.__dict__.keys():
                self.__dict__[key] = value
        
        num_dims = coarse_rgbs.ndim
        assert coarse_rgbs.ndim in [4, 5] 
        assert coarse_sigmas.ndim == coarse_rgbs.ndim
        assert pts_z.ndim == coarse_rgbs.ndim
        assert ray_origins.ndim == pts_z.ndim - 1 and ray_dirs.ndim == pts_z.ndim - 1

        if num_dims == 4:
            batch_size, num_rays, num_steps = coarse_rgbs.shape[:3]
        else:
            batch_size, H, W, num_steps = coarse_rgbs.shape[:4]
            num_rays = H * W
            coarse_rgbs = coarse_rgbs.reshape(batch_size, num_rays, num_steps, coarse_rgbs.shape[-1])
            coarse_sigmas = coarse_sigmas.reshape(batch_size, num_rays, num_steps, coarse_sigmas.shape[-1])
            pts_z = pts_z.reshape(batch_size, num_rays, num_steps, pts_z.shape[-1])
            ray_origins = ray_origins.reshape(batch_size, num_rays, -1)
            ray_dirs = ray_dirs.reshape(batch_size, num_rays, -1)

        # Get the importance of all points
        renderer_results = self.nerf_weight(sigmas=coarse_sigmas,
                                          pts_z=pts_z,
                                          noise_std=noise_std,
                                          last_back=self.last_back,
                                          white_back=self.white_back,
                                          clamp_mode=self.clamp_mode,
                                          render_mode=self.render_mode,
                                          fill_mode=self.fill_mode,
                                          max_depth=self.max_depth,
                                          num_per_group=self.num_per_group)
        # ret_weight = renderer_results['weights']
        weights = renderer_results.reshape(batch_size * num_rays, num_steps) + 1e-5

        # Importance sampling
        pts_z = pts_z.reshape(batch_size * num_rays, num_steps)
        pts_z_mid = 0.5 * (pts_z[:, :-1] + pts_z[:, 1:])
        if sample_model == 1:
            num_steps = num_steps
        elif sample_model == 0:
            num_steps = num_steps*2
        else:
            num_steps = num_steps//5
        fine_pts_z = self.sample_pdf(pts_z_mid,
                                      weights[:, 1:-1],
                                      num_steps,
                                      det=False).detach().reshape(batch_size, num_rays, num_steps, 1)
        fine_pts = ray_origins.unsqueeze(2).contiguous() + ray_dirs.unsqueeze(2).contiguous() * fine_pts_z.contiguous()

        if num_dims == 4: 
            fine_pts = fine_pts.reshape(batch_size, num_rays, num_steps, 3)                     
            fine_pts_z = fine_pts_z.reshape(batch_size, num_rays, num_steps, 1)
            ray_dirs = ray_dirs.reshape(batch_size, num_rays, 3)
        else:
            fine_pts = fine_pts.reshape(batch_size, H, W, num_steps, 3) 
            fine_pts_z = fine_pts_z.reshape(batch_size, H, W, num_steps, 1)
            ray_dirs = ray_dirs.reshape(batch_size, H, W, 3)
        results = {
            'pts': fine_pts,
            'pts_z': fine_pts_z,
            'ray_dirs': ray_dirs,
        }

        return results #, ret_weight.reshape(batch_size, H, W, num_steps, 1) 

    def nerf_weight(self,
                sigmas,
                pts_z,
                noise_std,
                last_back=False,
                white_back=False,
                clamp_mode=None,
                render_mode=None,
                fill_mode=None,
                max_depth=None,
                num_per_group=None):
        """ Integrate the values along the ray.

        Args:
            rgbs: （batch_size, H, W, num_steps, 3) or (batch_size, num_rays, num_steps, 3)
            sigmas: （batch_size, H, W, num_steps, 1) or (batch_size, num_rays, num_steps, 1)
            pts_z: (batch_size, H, W, num_steps, 1) or (batch_size, num_rays, num_steps, 1)
            noise_std: 
        
        Returns:
            rgb: (batch_size, H, W, 3) or (batch_size, num_rays, 3)
            depth: (batch_size, H, W, 1) or (batch_size, num_rays, 1)
            weights: (batch_size, H, W, num_steps, 1) or (batch_size, num_rays, num_steps, 1)
            alphas: (batch_size, H, W, 1) or (batch_size, num_rays, 1)
        """
        num_dims = sigmas.ndim
        assert num_dims in [4, 5]
        assert num_dims == pts_z.ndim

        if num_dims == 4:
            batch_size, num_rays, num_steps = sigmas.shape[:3]
        else:
            batch_size, H, W, num_steps = sigmas.shape[:4]
            sigmas = sigmas.reshape(batch_size, H * W, num_steps, sigmas.shape[-1])
            pts_z = pts_z.reshape(batch_size, H * W, num_steps, pts_z.shape[-1])

        if num_per_group is None:
            num_per_group = num_steps
        assert num_steps % num_per_group== 0

        # Get deltas for rendering.
        deltas = pts_z[:, :, 1:] - pts_z[:, :, :-1]
        if max_depth is not None:
            delta_inf = max_depth - pts_z[:, :, -1:]
        else:
            delta_inf = 1e10 * torch.ones_like(deltas[:, :, :1])
        deltas = torch.cat([deltas, delta_inf], -2)
        if render_mode == 'no_dist':
            deltas[:] = 1
        
        # Get alpha
        noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std
        if clamp_mode == 'softplus':
            alphas = 1-torch.exp(-deltas * (F.softplus(sigmas + noise)))
        elif clamp_mode == 'relu':
            alphas = 1 - torch.exp(-deltas * (F.relu(sigmas + noise)))
        elif clamp_mode == 'mipnerf':
            alphas = 1 - torch.exp(-deltas * (F.softplus(sigmas + noise - 1)))
        else:
            raise ValueError(f'Invalid clamping mode: `{clamp_mode}`!\n'
                            f'Types allowed: {list(_CLAMP_MODE)}.')

        # Get accumulated alphas
        alphas = alphas.reshape(alphas.shape[:2] + (num_steps//num_per_group, num_per_group, alphas.shape[-1])) 
        alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :, :1]), 1 - alphas + 1e-10], -2) 
        cum_alphas_shifted = torch.cumprod(alphas_shifted, -2)

        # Get weights
        weights = alphas * cum_alphas_shifted[:, :, :, :-1]
        weights_sum = weights.sum(3)

        pts_z = pts_z.reshape(pts_z.shape[:2] + (num_steps//num_per_group, num_per_group, pts_z.shape[-1])) 

        if last_back:
            weights[:, :, :, -1] += (1 - weights_sum)

        weights_final = weights.reshape(alphas.shape[:2]+(num_steps, 1))
        if num_dims == 5:
            weights_final = weights_final.reshape(batch_size, H, W, num_steps, 1)
        else:
            weights_final = weights_final.reshape(batch_size, num_rays, num_steps, 1)
        
        return weights_final

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """Sample @N_importance samples from @bins with distribution defined by @weights.

        Args:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero

        Returns:
            samples: the sampled samples
        Source: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py

        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps  # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1,
                                keepdim=True)  # (N_rays, N_samples_)
        cdf = torch.cumsum(
            pdf, -1)  # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf],
                        -1)  # (N_rays, N_samples_+1)
        # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u)
        below = torch.clamp_min(inds - 1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above],
                                -1).view(N_rays, 2 * N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled)
        cdf_g = cdf_g.view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0, in which case it will not be sampled
        # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] -
                                                                bins_g[..., 0])
        return samples


if __name__ == '__main__':
    hiesampler = HierarchicalSampling(last_back=False,
                 white_back=False,
                 clamp_mode='softplus',
                 render_mode=None,
                 fill_mode='weight',
                 max_depth=None,
                 num_per_group=None)
    device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    H = W = 64
    num_rays = 64*64
    num_steps = 24

    coarse_rgbs = torch.rand((batch_size, num_rays, num_steps, 3), device=device)
    coarse_sigmas = torch.rand((batch_size, num_rays, num_steps, 1), device=device)
    pts_z = torch.rand((batch_size, num_rays, num_steps, 1), device=device)
    ray_origins = torch.rand((batch_size, num_rays, 3), device=device)
    ray_dirs = torch.rand((batch_size, num_rays, 3), device=device)

    results = hiesampler(coarse_rgbs=coarse_rgbs,
                         coarse_sigmas=coarse_sigmas,
                         pts_z=pts_z,
                         ray_origins=ray_origins,
                         ray_dirs=ray_dirs)

    print(results['pts'][0,0,0])
    print(results['pts'][0,0,1])
    print(results['pts'][0,0,2])
    print(results['pts_z'][0,0,0])

