"""One pair per microbatch and a mean over complete authentication groups."""

import torch.nn.functional as F
from trustjev.losses.consistency import symmetric_kl_per_view
from trustjev.losses.authentication import paired_auth_ranking


def objective(first, second, labels, options):
    ce = 0.5 * (F.cross_entropy(first, labels) + F.cross_entropy(second, labels))
    consistency = symmetric_kl_per_view(first, second).mean()
    rank = paired_auth_ranking(
        first,
        second,
        labels,
        options.index("Access Granted"),
        options.index("Access Denied"),
        margin=1.0,
    )
    return ce + consistency + rank, {"ce": ce, "stochastic_kl": consistency, "paired_rank": rank}
