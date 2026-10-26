"""Policy-sufficient candidate separation against the evidence budget."""

import torch
import torch.nn.functional as F


def policy_sufficient_margin(out, labels, requires_evidence, slack=1.0):
    """Return per-view losses; evidence-required views have exactly zero loss.

    This objective uses training annotations, not an inference-time oracle.
    A correct authority margin above 2 beta is the sufficient stability
    condition already established for the isolated bounded scorer.
    """
    authority = out["authority"]
    valid = out["candidate_valid"].bool()
    beta = out["beta"].reshape(-1)
    assert authority.shape == valid.shape and labels.shape == beta.shape
    assert requires_evidence.shape == labels.shape and slack > 0
    assert (valid.sum(-1) >= 2).all()
    rows = torch.arange(len(labels), device=labels.device)
    assert valid[rows, labels].all()
    competitors = valid.clone()
    competitors[rows, labels] = False
    other = authority.masked_fill(~competitors, -torch.inf).max(-1).values
    margin = authority[rows, labels] - other
    loss = F.softplus(2 * beta + slack - margin)
    return loss * (~requires_evidence.bool()).to(loss.dtype)
