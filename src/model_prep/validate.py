"""Model validation utility (Milestone 3, Section 6.B).

Validates whether a model artifact (local directory or hub ref) meets the
architectural and metadata requirements of the requested OpenSI runtime backend.
It never modifies source files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .inspect import inspect_model, ModelInspection

SUPPORTED_TRANSFORMERS_ARCHITECTURES = {
    "Qwen3MoeForCausalLM",
    "Qwen2MoeForCausalLM",
    "Qwen2ForCausalLM",
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
}

SUPPORTED_BACKENDS = {"transformers", "llama_cpp", "mock", "openai_compatible"}


@dataclass
class ValidationResult:
    model_ref: str
    backend: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    architecture: str | None = None
    is_moe: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_ref": self.model_ref,
            "backend": self.backend,
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "architecture": self.architecture,
            "is_moe": self.is_moe,
            "details": self.details,
        }


def validate_model(
    model_ref: str,
    backend: str = "transformers",
    inspection: ModelInspection | None = None,
) -> ValidationResult:
    """Validate a model artifact against target backend constraints."""
    if backend not in SUPPORTED_BACKENDS:
        return ValidationResult(
            model_ref=str(model_ref),
            backend=backend,
            valid=False,
            errors=[f"Unsupported backend '{backend}'. Choose from: {sorted(SUPPORTED_BACKENDS)}"],
        )

    errors: list[str] = []
    warnings: list[str] = []

    if inspection is None:
        try:
            inspection = inspect_model(model_ref)
        except Exception as exc:
            return ValidationResult(
                model_ref=str(model_ref),
                backend=backend,
                valid=False,
                errors=[f"Failed to inspect model: {exc}"],
            )

    errors.extend(inspection.errors)

    # Basic architecture checks
    arch = inspection.architecture
    if not arch:
        errors.append("Model configuration missing architecture definition ('architectures' field).")

    if not inspection.num_hidden_layers or inspection.num_hidden_layers <= 0:
        errors.append("Invalid or missing 'num_hidden_layers'. Must be > 0.")

    if not inspection.hidden_size or inspection.hidden_size <= 0:
        errors.append("Invalid or missing 'hidden_size'. Must be > 0.")

    # MoE checks
    if inspection.is_moe():
        if not inspection.num_experts or inspection.num_experts <= 1:
            errors.append(f"MoE model declared but invalid 'num_experts': {inspection.num_experts}")
        if not inspection.num_experts_per_tok or inspection.num_experts_per_tok <= 0:
            errors.append(
                f"MoE model declared but invalid 'num_experts_per_tok': {inspection.num_experts_per_tok}"
            )
        if (
            inspection.num_experts
            and inspection.num_experts_per_tok
            and inspection.num_experts_per_tok > inspection.num_experts
        ):
            errors.append("num_experts_per_tok exceeds total num_experts.")

    # Context window check
    if not inspection.max_position_embeddings:
        warnings.append("max_position_embeddings is not explicitly defined in config.")
    elif inspection.max_position_embeddings < 2048:
        warnings.append(
            f"Context length {inspection.max_position_embeddings} is small for recurrent task loops (<2048)."
        )

    # Chat template check
    if not inspection.chat_template_present:
        warnings.append("No chat template discovered; structured prompting may require manual formatting.")

    # Backend-specific checks
    file_names = [f["name"] for f in inspection.files]

    if backend == "transformers":
        if arch and arch not in SUPPORTED_TRANSFORMERS_ARCHITECTURES:
            warnings.append(
                f"Architecture '{arch}' not in known tested set ({sorted(SUPPORTED_TRANSFORMERS_ARCHITECTURES)}). "
                "Transformers backend may require a custom loader or updated transformers release."
            )
        has_weights = any(
            name.endswith((".safetensors", ".bin")) for name in file_names
        )
        if inspection.source == "local" and not has_weights:
            errors.append("No weights files (.safetensors or .bin) found in model directory.")

    elif backend == "llama_cpp":
        has_gguf = any(name.endswith(".gguf") for name in file_names)
        if inspection.source == "local" and not has_gguf:
            warnings.append(
                "No .gguf file detected directly in model directory. Ensure conversion to GGUF is performed "
                "before serving with llama-server."
            )

    valid = len(errors) == 0
    return ValidationResult(
        model_ref=str(model_ref),
        backend=backend,
        valid=valid,
        errors=errors,
        warnings=warnings,
        architecture=arch,
        is_moe=inspection.is_moe(),
        details={
            "source": inspection.source,
            "num_hidden_layers": inspection.num_hidden_layers,
            "hidden_size": inspection.hidden_size,
            "num_experts": inspection.num_experts,
            "num_experts_per_tok": inspection.num_experts_per_tok,
            "max_position_embeddings": inspection.max_position_embeddings,
            "file_count": len(inspection.files),
        },
    )
