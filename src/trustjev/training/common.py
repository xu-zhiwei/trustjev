"""Single-device training, checkpoint selection and run provenance."""

from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time
import importlib.metadata
import numpy as np
import torch
from trustjev.checkpoint import write_json, file_hash, save_checkpoint, warm_start
from trustjev.data.batching import batch
from trustjev.data.loading import load_groups, encode_groups, check_split_overlap
from trustjev.inference import evaluate, dataset_metrics
from trustjev.modeling import ProvenanceDecisionModel, load_tokenizer
from . import stage1, stage2, authentication
from .adversarial import load_validator, prepare_bank, worst_case_evidence


def configure_runtime():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False


def optimizer_and_schedule(model, config, groups):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    steps = math.ceil(groups / config.group_batch) * config.epochs
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1, (s + 1) / max(1, int(0.05 * steps))) * max(0.1, 1 - s / steps)
    )
    return params, optimizer, schedule


def training_step(model, records, pad, source_prob, config, optimizer, device="cuda"):
    optimizer.zero_grad(set_to_none=True)
    values = []
    if config.stage == "authentication":
        for record in records:
            inputs, labels, *_ = batch([record], pad, device)
            first = model(inputs)["logits"]
            second = model(inputs)["logits"]
            loss, parts = authentication.objective(
                first, second, labels, record["views"][0]["options"]
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite authentication loss")
            (loss / len(records)).backward()
            values.append(
                {
                    "loss": float(loss.detach()),
                    **{name: float(value.detach()) for name, value in parts.items()},
                }
            )
    else:
        inputs, labels, required, slices = batch(records, pad, device)
        out = model(inputs)
        # Use independent dropout draws for the two prediction passes.
        second = model(inputs)["logits"] if config.stage == "stage2" else None
        if config.adversarial_weight or config.adversarial_invariance_weight:
            adv = worst_case_evidence(model, records, slices, out, pad, source_prob, device)
        else:
            adv = (out["logits"].sum() * 0, out["logits"].sum() * 0)
        if config.stage == "stage1":
            loss, parts = stage1.objective(
                out, labels, required, slices, records, source_prob, config, adv
            )
        else:
            loss, parts = stage2.objective(
                out, second, labels, required, slices, records, source_prob, config, adv
            )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite task-training loss")
        if config.stage == "stage2":
            stage2.backward(loss, out, required, slices, records, source_prob, config, model)
        else:
            loss.backward()
        values.append(
            {
                "loss": float(loss.detach()),
                **{name: float(value.detach()) for name, value in parts.items()},
            }
        )
    return {key: float(np.mean([value[key] for value in values])) for key in values[0]}


def _input_hashes(paths):
    result = {}
    for filename in paths:
        path = Path(filename)
        result[str(path)] = file_hash(path)
        sidecar = path.with_name(path.stem + "_eligibility.json")
        if sidecar.exists():
            result[str(sidecar)] = file_hash(sidecar)
    return result


def run_training(
    config,
    train_paths,
    development_paths,
    output,
    warm=None,
    validator_reference=None,
    device="cuda",
):
    config.validate()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for full-model training")
    if (config.stage != "stage1") != (warm is not None):
        raise ValueError("Stage 1 starts from the backbone, later stages need --warm-start")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output exists, choose a new run directory {output}")
    is_auth = config.stage == "authentication"
    train = load_groups(train_paths, authentication=is_auth)
    development = load_groups(development_paths, authentication=is_auth)
    check_split_overlap(train, development)
    if not is_auth and {r["dataset"] for r in train} != {r["dataset"] for r in development}:
        raise ValueError("Task-training development sources need to match training sources")
    validator = None
    if not is_auth and (config.adversarial_weight or config.adversarial_invariance_weight):
        validator = load_validator(validator_reference)
    configure_runtime()
    token = load_tokenizer(config.model)
    model = ProvenanceDecisionModel(config.model).to(device)
    upstream = warm_start(model, warm, config) if warm else None
    if config.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    encode_groups(train, token, config.model)
    encode_groups(development, token, config.model)
    attack_info = prepare_bank(train, token, config.model, validator) if validator else None
    counts = Counter(r["dataset"] for r in train)
    source_prob = {dataset: count / len(train) for dataset, count in counts.items()}
    params, optimizer, schedule = optimizer_and_schedule(model, config, len(train))
    output.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parents[1]
    provenance = {
        "config": config.to_dict(),
        "train_groups": len(train),
        "development_groups": len(development),
        "train_views": sum(len(r["views"]) for r in train),
        "source_probabilities": source_prob,
        "input_hashes": _input_hashes([*train_paths, *development_paths]),
        "warm_start_hashes": upstream,
        "validator": validator_reference,
        "adversarial_bank": attack_info,
        "source_hashes": {
            str(p.relative_to(source_root)): file_hash(p) for p in sorted(source_root.rglob("*.py"))
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "numpy", "safetensors", "PyYAML")
        },
        "device": device,
        "trainable_parameters": sum(p.numel() for p in params),
    }
    if device.startswith("cuda"):
        provenance["gpu"] = torch.cuda.get_device_name(torch.device(device))
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    write_json(output / "provenance.json", provenance)
    history, selection = [], {}
    started = time.monotonic()
    for epoch in range(config.epochs):
        model.train()
        order = list(range(len(train)))
        random.shuffle(order)
        values = []
        for offset in range(0, len(order), config.group_batch):
            records = [train[index] for index in order[offset : offset + config.group_batch]]
            value = training_step(
                model, records, token.pad_token_id, source_prob, config, optimizer, device
            )
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            schedule.step()
            values.append(value)
            if offset % (config.group_batch * 40) == 0:
                print(json.dumps({"epoch": epoch + 1, "groups": offset, **value}), flush=True)
        predictions = evaluate(
            model, development, token.pad_token_id, group_batch=1 if is_auth else 2, device=device
        )
        by_dataset = dataset_metrics(predictions)
        if any(m["balanced_score"] is None for m in by_dataset.values()):
            raise ValueError("Checkpoint selection needs attack and non-attack development views")
        snapshot = output / "epochs" / f"epoch-{epoch + 1}"
        metadata = {
            "stage": config.stage,
            "epoch": epoch + 1,
            "model": asdict(config.model),
            "development": by_dataset,
        }
        save_checkpoint(model, snapshot, metadata)
        write_json(snapshot / "development_predictions.json", predictions)
        for dataset, result in by_dataset.items():
            previous = selection.get(dataset)
            if is_auth or previous is None or result["balanced_score"] > previous["balanced_score"]:
                selection[dataset] = {
                    "epoch": epoch + 1,
                    "checkpoint": str(snapshot),
                    "balanced_score": result["balanced_score"],
                    "rule": "last authentication epoch"
                    if is_auth
                    else "maximum development balance, earliest tie",
                }
        history.append(
            {
                "epoch": epoch + 1,
                "development": by_dataset,
                "training": {key: float(np.mean([v[key] for v in values])) for key in values[0]},
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        write_json(output / "history.json", history)
        write_json(output / "selection.json", selection)
    result = {"epochs": config.epochs, "elapsed_seconds": time.monotonic() - started}
    if device.startswith("cuda"):
        result["peak_memory_gib"] = torch.cuda.max_memory_allocated(torch.device(device)) / 2**30
    write_json(output / "complete.json", result)
    return selection
