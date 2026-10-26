"""Evaluate a checkpoint with a fixed decision rule on supplied labeled groups."""

import argparse
from pathlib import Path
from trustjev.checkpoint import load_checkpoint, write_json
from trustjev.modeling import load_tokenizer
from trustjev.data.loading import load_groups, encode_groups
from trustjev.inference import evaluate, dataset_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--group-batch", type=int, default=2)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Output exists {output}")
    model = load_checkpoint(args.checkpoint, device=args.device, base_model=args.base_model)
    token = load_tokenizer(model.config)
    records = encode_groups(load_groups(args.data), token, model.config)
    rows = evaluate(
        model, records, token.pad_token_id, args.group_batch, args.device, args.threshold
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "predictions.json", rows)
    write_json(
        output / "metrics.json", {"datasets": dataset_metrics(rows), "threshold": args.threshold}
    )


if __name__ == "__main__":
    main()
