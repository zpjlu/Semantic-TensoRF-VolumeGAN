# python3.7
"""Collects all functions for rendering."""
from .points_sampling import PointsSampling
from .hierarchicle_sampling import HierarchicalSampling
from .renderer import Renderer
from .utils import interpolate_feature, interpolate_feature_triplane, interpolate_feature_3d, interpolate_feature_2d
from .nerf import NeRFSynthesisNetwork
__all__ = ['PointsSampling', 'HierarchicalSampling', 'Renderer', 'interpolate_feature']
