# Copyright (c) 2024-2025 Walter Calmels — TUCH Systems Research Laboratory
# All rights reserved.
# Non-commercial use only under LICENSE A. Commercial use requires LICENSE B.
# Contact: walter@tuch.systems | github.com/wcalmels/consciousai
# DOI: https://doi.org/10.5281/zenodo.20602077

"""
ConsciousAI Algorithm Suite
============================

Five algorithms derived from spectral integration theory,
applicable independently of IIT/consciousness framing.

Algorithms
----------
SpectralChangePoint  — regime change detection via rolling eigenvalue entropy
IntegrationDistance  — distance metric between systems (KL, JSD, Wasserstein)
CausalEmergence      — measures whole > sum of parts; component importance
TemporalIntegration  — multi-scale Φ̃ spectrogram (wavelet-like)
AdaptiveThreshold    — dynamic alert thresholds from Φ̃ history distribution

Quick start
-----------
>>> from src.algorithms import SpectralChangePoint, IntegrationDistance
>>> from src.algorithms import CausalEmergence, TemporalIntegration, AdaptiveThreshold
"""

from .change_point        import SpectralChangePoint, ChangePoint, CPDResult
from .integration_distance import IntegrationDistance, DistanceResult
from .causal_emergence    import CausalEmergence, EmergenceResult
from .temporal_integration import TemporalIntegration, TemporalIntegrationResult
from .adaptive_threshold  import AdaptiveThreshold, Alert, ThresholdResult

__all__ = [
    "SpectralChangePoint",  "ChangePoint",           "CPDResult",
    "IntegrationDistance",  "DistanceResult",
    "CausalEmergence",      "EmergenceResult",
    "TemporalIntegration",  "TemporalIntegrationResult",
    "AdaptiveThreshold",    "Alert",                 "ThresholdResult",
]
