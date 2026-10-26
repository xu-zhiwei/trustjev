"""Two stochastic passes and source-balanced authority supervision."""

import torch.nn.functional as F
from trustjev.data.batching import balanced_group_mean
from trustjev.losses.authority import (
    group_authority_supervision,
    group_uniformity,
    head_only_uniformity_backward,
)
from trustjev.losses.consistency import symmetric_kl_per_view
from trustjev.losses.margin import policy_sufficient_margin
from trustjev.losses.paired import paired_objective


def objective(out, second, labels, required, slices, records, source_prob, config, adversarial):
    logits = out["logits"]
    ce = balanced_group_mean(
        F.cross_entropy(logits, labels, reduction="none"), slices, records, source_prob
    )
    second_ce = balanced_group_mean(
        F.cross_entropy(second, labels, reduction="none"), slices, records, source_prob
    )
    consistency = balanced_group_mean(
        symmetric_kl_per_view(logits, second), slices, records, source_prob
    )
    inv, change = paired_objective(logits, labels, slices, records)
    authority = group_authority_supervision(out, labels, required, slices, records, source_prob)
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
        0.5 * (ce + second_ce)
        + consistency
        + config.margin_weight * margin
        + adversarial_loss
        + config.invariance_weight * inv
        + config.counterfactual_weight * change
        + config.authority_weight * authority
    )
    return loss, {
        "ce": ce,
        "second_ce": second_ce,
        "stochastic_kl": consistency,
        "invariance": inv,
        "counterfactual": change,
        "authority": authority,
        "policy_margin": margin,
        "adversarial_ce": adv_ce,
        "adversarial_invariance": adv_inv,
    }


def backward(loss, out, required, slices, records, source_prob, config, model):
    uniform = group_uniformity(out, required, slices, records, source_prob)
    weighted = config.authority_weight * uniform
    head_only_uniformity_backward(loss - weighted, weighted, model.authority.parameters())
