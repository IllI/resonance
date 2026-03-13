"""
D-LinOSS: Damped Linear Oscillatory State-Space Models
=======================================================
Local reference copy from: https://github.com/jaredbmit/damped-linoss

Contains:
- reference/    : Annotated mirror of the JAX/Equinox reference implementation
- dlinoss_torch.py : PyTorch port of D-LinOSS layers
- mri_artifact_correction.py : Forward model & inverse correction of MRI observation artifacts
- test_dlinoss.py : Comprehensive test suite validating mathematical soundness

The MRI helpers are imported lazily so code paths that only need the core
PyTorch D-LinOSS model do not eagerly pull in the wider SciPy MRI stack.
"""

from .dlinoss_torch import DLinOSSBlock, DLinOSSLayer, DLinOSSModel

__all__ = [
    "DLinOSSLayer",
    "DLinOSSBlock",
    "DLinOSSModel",
    "MRIArtifactCorrector",
    "MRIForwardModel",
]


def __getattr__(name: str):
    if name in {"MRIArtifactCorrector", "MRIForwardModel"}:
        from .mri_artifact_correction import MRIArtifactCorrector, MRIForwardModel

        exports = {
            "MRIArtifactCorrector": MRIArtifactCorrector,
            "MRIForwardModel": MRIForwardModel,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
