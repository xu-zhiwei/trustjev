"""First-stage objective with shared authority uniformity gradients."""

import torch.nn.functional as F
from trustjev.data.batching import balanced_group_mean
from trustjev.losses.authority import authority_supervision
from trustjev.losses.margin import policy_sufficient_margin
from trustjev.losses.paired import paired_objective


def objective(out, labels, required, slices, records, source_prob, config, adversarial):
    logits = out["logits"]
    ce = balanced_group_mean(
        F.cross_entropy(logits, labels, reduction="none"), slices, records, source_prob
    )
    inv, change = paired_objective(logits, labels, slices, records)
    authority = authority_supervision(out, labels, required)
    margin = balanced_group_mean(
        policy_sufficient_margin(out, labels, required, config.margin_slack),
        slices,
        records,
        source_prob,
    )
    adv_ce, adv_inv = adversarial
    adversarial_loss = (
        config.adversarial_weight * adv_ce + config.adversarial_invariance_weight * adv_inv
    )
    loss = (
        ce
        + config.margin_weight * margin
        + adversarial_loss
        + config.invariance_weight * inv
        + config.counterfactual_weight * change
        + config.authority_weight * authority
    )
    return loss, {
        "ce": ce,
        "invariance": inv,
        "counterfactual": change,
        "authority": authority,
        "policy_margin": margin,
        "adversarial_ce": adv_ce,
        "adversarial_invariance": adv_inv,
    }
