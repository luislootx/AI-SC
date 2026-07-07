"""Faithful baselines for honest comparison against the swarm-discovered hybrid.

These are NOT the simplified `branch_trunk` block in the swarm registry.
Each baseline here follows its original paper's architecture:

  - PureFNO         — Li et al. 2021 spectral conv blocks
  - DeepONetFaithful — Lu et al. 2021 with branch CNN + trunk MLP over query coords
  - PODDeepONet      — DeepONet with POD basis trunk (Lu et al. 2022)
  - PureTransformer  — multi-head spatial attention stack
"""
from .pure_fno import PureFNO
from .deeponet_faithful import DeepONetFaithful
from .pod_deeponet import PODDeepONet

__all__ = ["PureFNO", "DeepONetFaithful", "PODDeepONet"]
