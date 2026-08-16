"""Counterfactual for issue #39: what would the advisor predict with CORRECT model metadata?

Phase 1 showed all three ``transformers-*`` presets land on the low-confidence
path — not because the box is offline, but because ``microsoft/deberta-v3-*``
publishes no ``model.safetensors``, so ``HfApi().model_info().safetensors`` is
``None`` and ``_hub_metadata`` substitutes a flat 350 M-param "large model"
default for every deberta checkpoint.

That matters for the verdict: a preset flagged OVER on a 350 M stand-in may be
perfectly feasible at its real size. This script re-runs preflight with
``resolve_model`` patched to report:

  * ``total_params`` counted from the architecture (instantiated on the ``meta``
    device, so nothing is downloaded or allocated), and
  * ``total_file_bytes`` restricted to the files a torch load actually pulls
    (excludes ``tf_model.h5`` and the discarded ELECTRA ``*.generator.bin``).

and prints predicted-vs-corrected side by side.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModelForSequenceClassification

from autointent import Dataset, setup_logging
from autointent._advisor import (
    HardwareProfile,
    detect_hardware,
    load_config,
    run_preflight,
    stats_from_dataset_obj,
)
from autointent._advisor import _hub as hub_mod
from autointent._advisor._hub import ModelMeta

setup_logging("ERROR", log_filename="phase1b.log")

_BYTES_PER_GB = 1024**3
# Files a torch/transformers load never reads. tf_model.h5 is the TensorFlow
# mirror of the same weights; *.generator.bin is the ELECTRA-style generator
# that deberta-v3 ships but discards at fine-tune time.
_NON_TORCH_SUFFIXES = ("tf_model.h5", ".generator.bin", ".msgpack", ".onnx", ".h5")

_PRESETS = ("transformers-heavy", "transformers-light", "transformers-no-hpo")


def _true_param_count(model_name: str, n_labels: int) -> int:
    """Exact parameter count without downloading weights (meta-device init)."""
    cfg = AutoConfig.from_pretrained(model_name, num_labels=n_labels)
    with torch.device("meta"):
        model = AutoModelForSequenceClassification.from_config(cfg)
    return sum(p.numel() for p in model.parameters())


def _torch_only_bytes(model_name: str) -> int:
    info = HfApi().model_info(model_name, files_metadata=True)
    return sum(
        s.size
        for s in (info.siblings or [])
        if s.size and not s.rfilename.endswith(_NON_TORCH_SUFFIXES)
    )


def _corrected_meta(model_name: str, n_labels: int) -> ModelMeta:
    original = hub_mod.resolve_model(model_name)
    params = _true_param_count(model_name, n_labels)
    return ModelMeta(
        name=model_name,
        total_params=params,
        weight_bytes_per_param=4,  # deberta-v3 ships fp32
        total_file_bytes=_torch_only_bytes(model_name),
        cached_locally=False,  # force the honest cold-disk prediction
        confidence="hub",
        hidden_size=original.hidden_size,
        n_layers=original.n_layers,
    )


def main() -> None:
    parser = argparse.ArgumentParser("phase1b_metadata_counterfactual")
    parser.add_argument("--dataset", default="DeepPavlov/banking77")
    parser.add_argument("--output", default="calibration_runs/phase1b_counterfactual.json")
    parser.add_argument(
        "--assume-hardware",
        metavar="VRAM_GB,RAM_GB",
        help=(
            "Skip GPU probing and use this profile instead (e.g. '5.67,15.03'). "
            "Creating a CUDA context costs ~300 MB of VRAM, which is not affordable "
            "while a real fit is running on the same 6 GB card — pass the numbers "
            "detect_hardware() already reported instead."
        ),
    )
    args = parser.parse_args()

    if args.assume_hardware:
        vram_s, ram_s = args.assume_hardware.split(",")
        hardware = HardwareProfile(
            accelerator="cuda",
            device_name="assumed (no CUDA context created)",
            vram_gb=float(vram_s),
            ram_gb=float(ram_s),
            free_disk_gb=100.0,
            cpu_count=8,
        )
    else:
        hardware = detect_hardware()
    dataset = Dataset.from_hub(args.dataset)
    stats = stats_from_dataset_obj(dataset)
    print(
        f"Hardware: {hardware.accelerator} {hardware.vram_gb:.2f} GB VRAM | "
        f"dataset n_samples={stats.n_samples} n_classes={stats.n_classes}\n"
    )

    corrected_cache: dict[str, ModelMeta] = {}
    results: list[dict[str, Any]] = []

    for preset in _PRESETS:
        cfg, _ = load_config(preset)
        model_name = cfg["search_space"][0]["search_space"][0]["classification_model_config"][0]["model_name"]

        baseline = run_preflight(copy.deepcopy(cfg), stats, hardware, preset_name=preset)
        orig_meta = hub_mod.resolve_model(model_name)

        if model_name not in corrected_cache:
            corrected_cache[model_name] = _corrected_meta(model_name, stats.n_classes)
        fixed = corrected_cache[model_name]

        # Patch the memoized resolver for the duration of the second preflight.
        real_resolver = hub_mod.resolve_model

        def patched(name: str, _fixed: ModelMeta = fixed, _target: str = model_name) -> ModelMeta:
            return _fixed if name == _target else real_resolver(name)

        # _resource.py reaches the resolver as ``_hub.resolve_model(...)``
        # (module-attribute access), so rebinding it here is enough.
        hub_mod.resolve_model = patched  # type: ignore[assignment]
        try:
            corrected = run_preflight(copy.deepcopy(cfg), stats, hardware, preset_name=preset)
        finally:
            hub_mod.resolve_model = real_resolver  # type: ignore[assignment]

        row = {
            "preset": preset,
            "model": model_name,
            "params_assumed_M": round(orig_meta.total_params / 1e6, 1),
            "params_true_M": round(fixed.total_params / 1e6, 1),
            "disk_assumed_gb": round(orig_meta.disk_gb, 2),
            "disk_torch_only_gb": round(fixed.disk_gb, 2),
            "as_run": {
                "vram_gb": round(baseline.resource.vram_gb, 2),
                "headroom": baseline.headroom.value,
                "is_feasible": baseline.is_feasible,
                "low_confidence": baseline.low_confidence,
                "disk_download_gb": round(baseline.resource.disk_download_gb, 2),
            },
            "corrected": {
                "vram_gb": round(corrected.resource.vram_gb, 2),
                "headroom": corrected.headroom.value,
                "is_feasible": corrected.is_feasible,
                "low_confidence": corrected.low_confidence,
                "disk_download_gb": round(corrected.resource.disk_download_gb, 2),
            },
            "vram_budget_gb": round(hardware.vram_gb, 2),
        }
        results.append(row)
        print(
            f"{preset:22s} {model_name}\n"
            f"   params : assumed {row['params_assumed_M']:>7.1f} M -> true {row['params_true_M']:>7.1f} M\n"
            f"   VRAM   : as-run  {row['as_run']['vram_gb']:>7.2f} GB ({row['as_run']['headroom']})"
            f" -> corrected {row['corrected']['vram_gb']:>7.2f} GB ({row['corrected']['headroom']})"
            f"   [budget {row['vram_budget_gb']:.2f} GB]\n"
            f"   disk   : as-run  {row['as_run']['disk_download_gb']:>7.2f} GB download"
            f" -> corrected {row['corrected']['disk_download_gb']:>7.2f} GB\n"
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"hardware": {"vram_gb": hardware.vram_gb}, "rows": results}, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
