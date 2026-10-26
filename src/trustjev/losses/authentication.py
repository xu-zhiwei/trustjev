"""Supervised legitimate-versus-attack ranking."""

import torch
import torch.nn.functional as F


def paired_auth_ranking(first, second, labels, grant_index, denied_index, margin=1.0):
    assert first.shape == second.shape and first.ndim == 2 and first.shape[0] == 2
    assert first.shape[1] == 2 and {grant_index, denied_index} == {0, 1}
    legitimate = torch.where(labels == grant_index)[0]
    attack = torch.where(labels == denied_index)[0]
    assert legitimate.numel() == attack.numel() == 1
    terms = []
    for logits in [first, second]:
        scores = logits[:, grant_index] - logits[:, denied_index]
        separation = scores[legitimate[0]] - scores[attack[0]]
        terms.append(F.softplus(margin - separation))
    return 0.5 * (terms[0] + terms[1])
