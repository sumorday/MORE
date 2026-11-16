"""Spherical max-margin + cross-step residual routing prototype (see the module docstrings).

  spherical_svm  common/differential split of a routed merge, K x K max-margin solver,
                 per-layer runtime factors
  curvature      per-layer crowding index from alpha_spec, adaptive margin / damping weights
  dynamic_hook   inference-time routing: timestep schedule, joint-attention maps, gated deltas
"""
import os
import sys

# the merge code (ssr_rc, tcpr, common) lives in scripts/ as flat modules
_SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
