"""Prediction consistency under input and dropout interventions."""

import torch.nn.functional as F


def sym_kl(z1, z2):
    l1 = z1.log_softmax(-1)
    l2 = z2.log_softmax(-1)
    return 0.5 * (
        F.kl_div(l1, l2, log_target=True, reduction="batchmean")
        + F.kl_div(l2, l1, log_target=True, reduction="batchmean")
    )


def symmetric_kl_per_view(first, second):
    a = first.log_softmax(-1)
    b = second.log_softmax(-1)
    return 0.5 * (
        F.kl_div(a, b, log_target=True, reduction="none").sum(-1)
        + F.kl_div(b, a, log_target=True, reduction="none").sum(-1)
    )
