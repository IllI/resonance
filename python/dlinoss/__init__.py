"""
D-LinOSS: Damped Linear Oscillatory State-Space Models
=======================================================
Local reference copy from: https://github.com/jaredbmit/damped-linoss

Contains:
- reference/    : Annotated mirror of the JAX/Equinox reference implementation
- dlinoss_torch.py : PyTorch port of D-LinOSS layers
- mri_artifact_correction.py : Forward model & inverse correction of MRI observation artifacts
- test_dlinoss.py : Comprehensive test suite validating mathematical soundness
"""
from .dlinoss_torch import DLinOSSLayer, DLinOSSBlock, DLinOSSModel
from .mri_artifact_correction import MRIArtifactCorrector, MRIForwardModel
