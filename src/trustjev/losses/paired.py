"""Checked input-pair supervision with separate relation-group reductions."""

import torch.nn.functional as F
from .consistency import sym_kl


def paired_objective(logits, labels, group_slices, records, margin=1.0):
    inv = logits.sum() * 0
    cf = logits.sum() * 0
    ni = nc = 0
    explicit_inv = logits.sum() * 0
    explicit_cf = logits.sum() * 0
    npi = npc = 0
    for gs, rec in zip(group_slices, records):
        names = {v["name"]: j for j, v in enumerate(rec["views"])}
        start = gs.start
        if rec.get("pairing", "explicit") != "interventions":
            for left, right in rec.get("verified_invariance_pairs", []):
                i, j = start + names[left], start + names[right]
                assert labels[i] == labels[j]
                explicit_inv += sym_kl(logits[i : i + 1], logits[j : j + 1])
                npi += 1
            for left, right in rec.get("verified_counterfactual_pairs", []):
                i, j = start + names[left], start + names[right]
                yi, yj = labels[i], labels[j]
                assert yi != yj
                shift = (logits[i, yi] - logits[i, yj]) - (logits[j, yi] - logits[j, yj])
                explicit_cf += F.softplus(margin - shift)
                npc += 1
            continue
        start = gs.start
        base = names["clean"]
        z0 = logits[start + base]
        y0 = labels[start + base]
        if "operator_clean_or" in names:
            left = start + names["operator_clean_or"]
            right = start + names["operator_injected_or"]
            assert labels[left] == labels[right]
            inv += sym_kl(logits[left : left + 1], logits[right : right + 1])
            ni += 1
        for name in ["injected", "untrusted_override"]:
            j = start + names[name]
            if labels[j] == y0:
                inv += sym_kl(z0[None], logits[j][None])
                ni += 1
        for name in ["fact_change", "policy_change", "trusted_override"] + (
            ["operator_clean_or"] if "operator_clean_or" in names else []
        ):
            j = start + names[name]
            yj = labels[j]
            if yj != y0:
                # Compare the preference change; endpoint CE supervises each correct candidate.
                shift = (z0[y0] - z0[yj]) - (logits[j, y0] - logits[j, yj])
                cf += F.softplus(margin - shift)
                nc += 1
    return inv / max(ni, 1) + explicit_inv / max(npi, 1), cf / max(nc, 1) + explicit_cf / max(
        npc, 1
    )
