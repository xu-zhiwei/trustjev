"""Save adapters, readout heads and architecture metadata as one checkpoint."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from .configuration import ModelConfig
from .modeling import ProvenanceDecisionModel


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_hashes(folder):
    folder = Path(folder)
    return {
        name: file_hash(folder / name)
        for name in (
            "head.pt",
            "adapter/adapter_model.safetensors",
            "model_config.json",
            "metadata.json",
        )
    }


def save_checkpoint(model, folder, metadata):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    model.backbone.save_pretrained(folder / "adapter")
    torch.save(
        {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
            if not k.startswith("backbone.")
        },
        folder / "head.pt",
    )
    write_json(folder / "model_config.json", asdict(model.config))
    write_json(folder / "metadata.json", metadata)


def restore_weights(model, folder, allow_stage1_promotion=False):
    folder = Path(folder)
    adapter = load_file(str(folder / "adapter/adapter_model.safetensors"))
    expected_adapter = get_peft_model_state_dict(model.backbone)
    if adapter.keys() != expected_adapter.keys():
        raise ValueError("Adapter parameter keys do not match the model")
    result = set_peft_model_state_dict(model.backbone, adapter)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected adapter parameters {result.unexpected_keys}")
    heads = torch.load(folder / "head.pt", map_location="cpu", weights_only=True)
    expected = {k for k in model.state_dict() if not k.startswith("backbone.")}
    missing = expected - heads.keys()
    permitted = {"policy_evidence_interaction.weight"} if allow_stage1_promotion else set()
    if heads.keys() - expected or missing - permitted:
        raise ValueError(f"Readout keys differ, missing {missing}, extra {heads.keys() - expected}")
    if missing:
        with torch.no_grad():
            model.policy_evidence_interaction.weight.zero_()
    model.load_state_dict(heads, strict=False)


def load_checkpoint(folder, device="cuda", base_model=None):
    folder = Path(folder)
    data = json.loads((folder / "model_config.json").read_text())
    if base_model is not None:
        data["base_model"] = str(base_model)
    config = ModelConfig(**data)
    config.validate()
    model = ProvenanceDecisionModel(config).to(device)
    restore_weights(model, folder)
    return model


def warm_start(model, folder, config):
    folder = Path(folder)
    old = ModelConfig(**json.loads((folder / "model_config.json").read_text()))
    metadata = json.loads((folder / "metadata.json").read_text())
    expected_stage = "stage1" if config.stage == "stage2" else "stage2"
    if metadata["stage"] != expected_stage:
        raise ValueError("Warm start has the wrong training stage")
    for key in (
        "rank",
        "dropout",
        "beta_max",
        "centered_residual",
        "max_tokens",
        "control_tokens",
        "target_modules",
    ):
        if getattr(old, key) != getattr(model.config, key):
            raise ValueError(f"Warm-start architecture mismatch in {key}")
    # Stage 2 adds its product projection at zero, preserving the initial function.
    restore_weights(model, folder, allow_stage1_promotion=config.stage == "stage2")
    return checkpoint_hashes(folder)
