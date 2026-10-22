"""OpenSI model_prep package.

Inspects model artifacts, validates backend compatibility, produces
manifests, and packages runtime configuration. It never modifies
original model files and never claims that conversion adds cognitive
capability (handoff Sections 6.B and 15).
"""

from .inspect import inspect_model, ModelInspection
from .manifest import write_manifest, read_manifest
from .validate import validate_model, ValidationResult
from .package import package_config

__all__ = [
    "inspect_model",
    "ModelInspection",
    "write_manifest",
    "read_manifest",
    "validate_model",
    "ValidationResult",
    "package_config",
]
