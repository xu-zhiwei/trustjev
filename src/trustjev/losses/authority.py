"""Stage-specific authority targets and authority-head gradient routing."""

import torch
import torch.nn.functional as F


def authority_supervision(out, labels, requires_evidence):
    a = out["authority"]
    known = ~requires_evidence
    loss = a.sum() * 0
    if known.any():
        loss = loss + F.cross_entropy(a[known], labels[known])
    if requires_evidence.any():
        aa = a[requires_evidence]
        valid = out["candidate_valid"][requires_evidence]
        u = valid.float() / valid.sum(-1, keepdim=True)
        loss = loss + (u * (u.clamp_min(1e-8).log() - aa.log_softmax(-1))).sum(-1).mean()
    return loss


def head_only_uniformity_backward(primary_loss, uniform_loss, authority_parameters):
    """Update the authority head with uniformity without changing shared features.

    Both objectives use the same forward pass and dropout realization. Policy-
    known CE belongs in primary_loss and retains its normal gradient path.
    """
    params = list(authority_parameters)
    calibration = torch.autograd.grad(uniform_loss, params, retain_graph=True, allow_unused=True)
    primary_loss.backward()
    with torch.no_grad():
        for p, g in zip(params, calibration):
            if g is None:
                continue
            if p.grad is None:
                p.grad = g.detach().clone()
            else:
                p.grad.add_(g)


def group_authority_supervision(out, labels, requires_evidence, slices, records, source_prob):
    """Equal parent weight, with separate known/evidence terms within each parent.

    Missing strata contribute zero, matching a per-parent expectation rather than
    renormalizing by the number of parents containing that stratum.
    """
    assert len(slices) == len(records) and records
    assert set(source_prob) == {r["dataset"] for r in records} or all(
        r["dataset"] in source_prob for r in records
    )
    terms = []
    for gs, record in zip(slices, records):
        authority = out["authority"][gs]
        valid = out["candidate_valid"][gs]
        required = requires_evidence[gs]
        target = labels[gs]
        term = authority.sum() * 0
        known = ~required
        if known.any():
            term = term + F.cross_entropy(authority[known], target[known])
        if required.any():
            selected = authority[required]
            mask = valid[required]
            assert mask.any(-1).all()
            uniform = mask.to(selected.dtype) / mask.sum(-1, keepdim=True)
            term = (
                term
                + (uniform * (uniform.clamp_min(1e-8).log() - selected.log_softmax(-1)))
                .sum(-1)
                .mean()
            )
        probability = source_prob[record["dataset"]]
        assert probability > 0
        terms.append(term / (len(source_prob) * probability))
    return torch.stack(terms).mean()


def group_uniformity(out, requires_evidence, slices, records, source_prob):
    assert len(slices) == len(records) and records
    terms = []
    for gs, record in zip(slices, records):
        authority = out["authority"][gs]
        selected = requires_evidence[gs]
        term = authority.sum() * 0
        if selected.any():
            scores = authority[selected]
            mask = out["candidate_valid"][gs][selected]
            assert mask.any(-1).all()
            uniform = mask.to(scores.dtype) / mask.sum(-1, keepdim=True)
            term = (
                (uniform * (uniform.clamp_min(1e-8).log() - scores.log_softmax(-1))).sum(-1).mean()
            )
        probability = source_prob[record["dataset"]]
        assert probability > 0
        terms.append(term / (len(source_prob) * probability))
    return torch.stack(terms).mean()
