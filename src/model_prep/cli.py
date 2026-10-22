"""Command-line interface for model_prep (Milestone 3, Section 6.B).

Commands:
  inspect           Inspect architecture facts of a local or hub model
  validate          Validate compatibility against a target runtime backend
  export-manifest   Generate reproducible manifest file
  package-config    Package runtime config and sidecar metadata
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .inspect import inspect_model
from .manifest import write_manifest
from .package import package_config
from .validate import validate_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-prep",
        description="OpenSI Model Preparation & Validation Utility",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect a model artifact or hub id")
    p_inspect.add_argument("model", help="Local directory path or Hugging Face repo id")
    p_inspect.add_argument("--json", action="store_true", help="Output raw JSON")

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate model for target backend")
    p_validate.add_argument("model", help="Local directory path or Hugging Face repo id")
    p_validate.add_argument(
        "--backend",
        default="transformers",
        choices=["transformers", "llama_cpp", "mock", "openai_compatible"],
        help="Target runtime backend",
    )
    p_validate.add_argument("--json", action="store_true", help="Output raw JSON")

    # export-manifest
    p_manifest = subparsers.add_parser("export-manifest", help="Export reproducible manifest")
    p_manifest.add_argument("model", help="Local directory path or Hugging Face repo id")
    p_manifest.add_argument("--out", "-o", required=True, help="Output directory")
    p_manifest.add_argument(
        "--profile", default="recurrent-prototype", help="Configuration profile name"
    )

    # package-config
    p_package = subparsers.add_parser("package-config", help="Package config and sidecar metadata")
    p_package.add_argument("model", help="Local directory path or Hugging Face repo id")
    p_package.add_argument("--out", "-o", required=True, help="Output directory")
    p_package.add_argument(
        "--profile", default="recurrent-prototype", help="Configuration profile name"
    )
    p_package.add_argument(
        "--backend", default="transformers", help="Target runtime backend"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        res = inspect_model(args.model)
        if args.json:
            print(json.dumps(res.to_dict(), indent=2))
        else:
            print(f"Model: {res.model_ref} (source: {res.source})")
            print(f"Architecture: {res.architecture or 'unknown'}")
            print(f"Type: {res.model_type or 'unknown'} | Is MoE: {res.is_moe()}")
            print(f"Layers: {res.num_hidden_layers} | Hidden size: {res.hidden_size}")
            if res.is_moe():
                print(f"Experts: {res.num_experts} (routed: {res.num_experts_per_tok}/token)")
            print(f"Max context: {res.max_position_embeddings}")
            print(f"Chat template present: {res.chat_template_present}")
            print(f"Files found: {len(res.files)}")
            if res.errors:
                print(f"Errors: {res.errors}", file=sys.stderr)
                return 1
        return 0

    elif args.command == "validate":
        res = validate_model(args.model, backend=args.backend)
        if args.json:
            print(json.dumps(res.to_dict(), indent=2))
        else:
            status = "VALID" if res.valid else "INVALID"
            print(f"[{status}] Model: {res.model_ref} for backend '{res.backend}'")
            if res.warnings:
                print("Warnings:")
                for w in res.warnings:
                    print(f"  - {w}")
            if res.errors:
                print("Errors:", file=sys.stderr)
                for e in res.errors:
                    print(f"  - {e}", file=sys.stderr)
                return 1
        return 0 if res.valid else 1

    elif args.command == "export-manifest":
        path = write_manifest(args.model, args.out, profile=args.profile)
        print(f"Manifest written to: {path}")
        return 0

    elif args.command == "package-config":
        files = package_config(
            args.model,
            args.out,
            profile=args.profile,
            backend=args.backend,
        )
        print("Packaging complete:")
        for k, p in files.items():
            print(f"  {k:15s}: {p}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
