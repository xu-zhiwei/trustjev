"""Application-owned boundaries, logical positions and isolated attention."""

import copy
import re
import torch

SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"]


def protected_prefix(encoded, return_order=False):
    n = len(encoded["candidate_idx"])
    segments = encoded["segments"]
    order = [i for i, s in enumerate(segments) if s == 0]
    order += [i for i, s in enumerate(segments) if 2 <= s < n + 2]
    order += [i for i, s in enumerate(segments) if s == 1]
    order += [i for i, s in enumerate(segments) if s >= n + 2]
    assert sorted(order) == list(range(len(order)))
    inverse = {old: new for new, old in enumerate(order)}
    out = copy.deepcopy(encoded)
    for key in ["ids", "segments", "positions"]:
        out[key] = [encoded[key][i] for i in order]
    out["policy_idx"] = inverse[encoded["policy_idx"]]
    for key in ["evidence_idx", "candidate_idx", "context_candidate_idx"]:
        out[key] = [inverse[i] for i in encoded[key]]
    if "lexical_lengths" in encoded:
        out["lexical_lengths"] = encoded["lexical_lengths"][order].copy()
    return (out, order) if return_order else out


def text_tokens(tok, text, control_tokens=None):
    markers = SPECIAL if control_tokens is None else control_tokens
    text = re.sub(r"<\|([^<>]+)\|>", r"<¦\1¦>", text)
    codepoint = 0x2400
    while chr(codepoint) in "".join(markers):
        codepoint += 1
    replacement = chr(codepoint)
    for marker in sorted(markers, key=len, reverse=True):
        escaped = marker.translate(str.maketrans({"<": "‹", ">": "›", "|": "¦"}))
        if escaped == marker or any(other in escaped for other in markers):
            escaped = replacement * len(marker)
        text = text.replace(marker, escaped)
    return tok(text, add_special_tokens=False)["input_ids"]


def encode(tok, v, max_tokens=1536, protected=True, control_tokens=None):
    control = [
        tok.convert_tokens_to_ids(t)
        for t in (SPECIAL if control_tokens is None else control_tokens)
    ]
    if (
        len(control) != 3
        or len(set(control)) != 3
        or not all(i is not None and i >= 0 and i != tok.unk_token_id for i in control)
    ):
        raise ValueError("The tokenizer needs three distinct existing control tokens")
    policy = [control[0]] + text_tokens(tok, v["policy"], control_tokens) + [control[2]]
    evidence = [control[1]] + text_tokens(tok, v["evidence"], control_tokens) + [control[2]]
    ids = policy + evidence
    segments = [0] * len(policy) + [1] * len(evidence)
    positions = list(range(len(policy))) + list(range(len(policy), len(policy) + len(evidence)))
    policy_idx = len(policy) - 1
    evidence_idx = list(range(len(policy), len(ids)))
    candidate_idx = []
    for k, option in enumerate(v["options"]):
        c = [control[1]] + text_tokens(tok, "Candidate: " + option, control_tokens) + [control[2]]
        ids += c
        segments += [k + 2] * len(c)
        # Same candidate positions regardless of evidence length or candidate order.
        positions += list(range(len(policy), len(policy) + len(c)))
        candidate_idx.append(len(ids) - 1)
    context_candidate_idx = []
    context_start = len(policy) + len(evidence)
    for k, option in enumerate(v["options"]):
        cc = [control[1]] + text_tokens(tok, "Candidate: " + option, control_tokens) + [control[2]]
        ids += cc
        segments += [len(v["options"]) + k + 2] * len(cc)
        positions += list(range(context_start, context_start + len(cc)))
        context_candidate_idx.append(len(ids) - 1)
    if len(ids) > max_tokens:
        raise ValueError(f"overlong:{len(ids)}>{max_tokens}")
    encoded = {
        "ids": ids,
        "segments": segments,
        "positions": positions,
        "policy_idx": policy_idx,
        "evidence_idx": evidence_idx,
        "candidate_idx": candidate_idx,
        "label": v.get("label"),
        "requires_evidence": v.get("requires_evidence"),
        "view": v.get("name", "request"),
        "context_candidate_idx": context_candidate_idx,
    }

    return protected_prefix(encoded) if protected else encoded


def batch_inputs(encs, pad, device="cuda"):
    b = len(encs)
    length = max(len(e["ids"]) for e in encs)
    k = max(len(e["candidate_idx"]) for e in encs)
    ids = torch.full((b, length), pad, dtype=torch.long, device=device)
    positions = torch.zeros((b, length), dtype=torch.long, device=device)
    segments = torch.full((b, length), -1, dtype=torch.long, device=device)
    candidates = torch.zeros((b, k), dtype=torch.long, device=device)
    valid = torch.zeros((b, k), dtype=torch.bool, device=device)
    evidence_mask = torch.zeros((b, length), dtype=torch.bool, device=device)
    policy_idx = []
    context_candidates = torch.zeros((b, k), dtype=torch.long, device=device)
    has_context = all(bool(e.get("context_candidate_idx")) for e in encs)
    for i, e in enumerate(encs):
        n = len(e["ids"])
        ids[i, :n] = torch.tensor(e["ids"], device=device)
        positions[i, :n] = torch.tensor(e["positions"], device=device)
        segments[i, :n] = torch.tensor(e["segments"], device=device)
        nk = len(e["candidate_idx"])
        candidates[i, :nk] = torch.tensor(e["candidate_idx"], device=device)
        valid[i, :nk] = True
        evidence_mask[i, e["evidence_idx"]] = True
        policy_idx.append(e["policy_idx"])
        if has_context:
            context_candidates[i, :nk] = torch.tensor(e["context_candidate_idx"], device=device)
    causal = torch.ones(length, length, device=device, dtype=torch.bool).tril()
    allowed = (
        causal[None]
        & ((segments[:, :, None] == segments[:, None, :]) | (segments[:, None, :] == 0))
        & (segments[:, None, :] >= 0)
    )
    if has_context:
        context_query = segments >= (valid.sum(-1)[:, None] + 2)
        allowed |= causal[None] & context_query[:, :, None] & (segments[:, None, :] == 1)
    allowed |= torch.eye(length, dtype=torch.bool, device=device)[None]
    # SDPA accepts an application-owned additive 4D attention mask.
    mask = torch.zeros((b, length, length), dtype=torch.bfloat16, device=device).masked_fill(
        ~allowed, torch.finfo(torch.bfloat16).min
    )[:, None]
    return {
        "input_ids": ids,
        "position_ids": positions,
        "attention_mask": mask,
        "policy_idx": torch.tensor(policy_idx, device=device),
        "candidate_idx": candidates,
        "candidate_valid": valid,
        "evidence_mask": evidence_mask,
        "context_candidate_idx": context_candidates if has_context else None,
    }
