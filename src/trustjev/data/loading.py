"""Load user-supplied labeled input groups."""

import json
from pathlib import Path
from trustjev.encoding import encode
from .schema import validate_group


def load_groups(paths, authentication=False):
    records, seen = [], set()
    for filename in paths:
        path = Path(filename)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        eligibility = path.with_name(path.stem + "_eligibility.json")
        if eligibility.exists():
            accepted = json.loads(eligibility.read_text())["accepted_ids"]
            if not isinstance(accepted, list) or len(set(accepted)) != len(accepted):
                raise ValueError(f"Invalid accepted_ids in {eligibility}")
            if set(accepted) - {row["id"] for row in rows}:
                raise ValueError(f"Eligibility IDs missing from {path}")
            rows = [row for row in rows if row["id"] in set(accepted)]
        for row in rows:
            validate_group(row, authentication=authentication)
            key = (row["dataset"], row["id"])
            if key in seen:
                raise ValueError(f"Duplicate group {key}")
            seen.add(key)
            records.append(row)
    if not records:
        raise ValueError("No eligible source groups")
    return records


def encode_groups(records, token, config):
    for row in records:
        row["encoded"] = [
            encode(
                token,
                view,
                max_tokens=config.max_tokens,
                protected=config.protected,
                control_tokens=config.control_tokens,
            )
            for view in row["views"]
        ]
    return records


def check_split_overlap(train, development):
    """Reject an identical declared group or cluster across train and development."""
    for key in ("id", "cluster"):
        left = {(r["dataset"], r[key]) for r in train if key in r}
        right = {(r["dataset"], r[key]) for r in development if key in r}
        if left & right:
            raise ValueError(f"Train/development overlap in {key}")
