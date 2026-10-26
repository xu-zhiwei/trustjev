"""Shared Transformer backbone with protected authority and bounded evidence scores."""

import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer


def bounded_residual(e, beta, valid, centered=False):
    if centered:
        # Softmax is insensitive to a common logit offset. Remove that degree
        # of freedom before applying the saturating bounded nonlinearity.
        center = (e * valid).sum(-1, keepdim=True) / valid.sum(-1, keepdim=True)
        e = e - center
    return beta * torch.tanh(e)


class ProvenanceDecisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        base = config.base_model
        rank, dropout = config.rank, config.dropout
        beta_max, context_mix = config.beta_max, config.context_mix
        centered_residual = config.centered_residual
        lm = AutoModel.from_pretrained(
            base,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=config.local_files_only,
        )
        lm.config.use_cache = False
        self.backbone = get_peft_model(
            lm,
            LoraConfig(
                r=rank,
                lora_alpha=2 * rank,
                lora_dropout=dropout,
                target_modules=config.target_modules,
                bias="none",
            ),
        )
        d = lm.config.hidden_size
        h = 128
        self.readout_norm = nn.LayerNorm(d)
        self.policy = nn.Linear(d, h)
        self.candidate = nn.Linear(d, h)
        self.key = nn.Linear(d, h)
        self.value = nn.Linear(d, h)
        self.authority = nn.Sequential(
            nn.Linear(3 * h, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, 1)
        )
        self.evidence = nn.Sequential(
            nn.Linear(3 * h, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, 1)
        )
        self.gate = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 1))
        # Preserve the parent RNG stream and initial function exactly.
        if config.interaction:
            with torch.random.fork_rng(devices=[]):
                self.policy_evidence_interaction = nn.Linear(h, h, bias=False)
                nn.init.zeros_(self.policy_evidence_interaction.weight)
        self.beta_max = beta_max
        self.context_mix = context_mix
        self.centered_residual = centered_residual
        assert 0 <= context_mix <= 1

    def forward(self, b):
        h = self.backbone(
            input_ids=b["input_ids"],
            position_ids=b["position_ids"],
            attention_mask=b["attention_mask"],
            use_cache=False,
        ).last_hidden_state.float()
        h = self.readout_norm(h)
        device = h.device
        n = h.shape[0]
        valid = b["candidate_valid"]
        p = torch.tanh(self.policy(h[torch.arange(n, device=device), b["policy_idx"]]))
        c = torch.tanh(
            self.candidate(h[torch.arange(n, device=device)[:, None], b["candidate_idx"]])
        )
        pp = p[:, None].expand_as(c)
        a = self.authority(torch.cat([pp, c, pp * c], -1)).squeeze(-1)
        key = self.key(h)
        value = self.value(h)
        att = torch.einsum("bkh,blh->bkl", c, key) / (c.shape[-1] ** 0.5)
        att = att.masked_fill(~b["evidence_mask"][:, None], -1e9).softmax(-1)
        pooled = torch.einsum("bkl,blh->bkh", att, value)
        if self.context_mix > 0:
            assert b.get("context_candidate_idx") is not None
            context_h = h[torch.arange(n, device=device)[:, None], b["context_candidate_idx"]]
            context_value = self.value(context_h)
            pooled = (1 - self.context_mix) * pooled + self.context_mix * context_value
        # Policy-conditioned evidence enters only the bounded evidence branch.
        evidence_hidden = self.evidence[0](torch.cat([pp, c, pooled], -1))
        if self.config.interaction:
            evidence_hidden = evidence_hidden + self.policy_evidence_interaction(pp * pooled)
        e = self.evidence[3](self.evidence[2](self.evidence[1](evidence_hidden))).squeeze(-1)
        mean_c = (c * valid[:, :, None]).sum(1) / valid.sum(1)[:, None]
        # A fixed small residual bound would cap the attainable confidence when K is large.
        # A logarithmic candidate-count budget preserves finite influence and enough decision capacity.
        cap = self.beta_max + 0.5 * valid.sum(-1).float().log()
        beta = cap[:, None] * torch.sigmoid(self.gate(torch.cat([p, mean_c], -1)))
        residual = bounded_residual(e, beta, valid, self.centered_residual)
        z = (a + residual).masked_fill(~valid, -1e9)
        a = a.masked_fill(~valid, -1e9)
        top = a.topk(min(2, a.shape[1]), dim=-1)
        margin = (
            top.values[:, 0] - top.values[:, 1]
            if a.shape[1] > 1
            else torch.full((n,), float("inf"), device=device)
        )
        certificate = margin > 2 * beta.squeeze(-1)
        return {
            "logits": z,
            "authority": a,
            "beta": beta.squeeze(-1),
            "residual": residual,
            "certificate": certificate,
            "candidate_valid": valid,
        }


def load_tokenizer(config):
    token = AutoTokenizer.from_pretrained(
        config.base_model, local_files_only=config.local_files_only
    )
    if token.pad_token_id is None:
        raise ValueError("The tokenizer needs an existing padding token")
    return token
