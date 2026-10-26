"""Keep complete source groups together during training."""

import torch
from trustjev.encoding import batch_inputs


def balanced_group_mean(per_view, slices, records, source_prob):
    assert len(slices) == len(records) and records
    assert all(source_prob[r["dataset"]] > 0 for r in records)
    return torch.stack(
        [
            per_view[gs].mean() / (len(source_prob) * source_prob[r["dataset"]])
            for r, gs in zip(records, slices)
        ]
    ).mean()


def batch(groups, pad, device="cuda"):
    encodings, labels, required, slices = [], [], [], []
    for record in groups:
        start = len(encodings)
        encodings.extend(record["encoded"])
        labels.extend(view["label"] for view in record["views"])
        required.extend(view["requires_evidence"] for view in record["views"])
        slices.append(slice(start, len(encodings)))
    inputs = batch_inputs(encodings, pad, device=device)
    return (
        inputs,
        torch.tensor(labels, device=device),
        torch.tensor(required, dtype=torch.bool, device=device),
        slices,
    )
