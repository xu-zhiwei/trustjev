"""Select one candidate from a JSON object containing policy, evidence and options."""

import argparse
import json
from pathlib import Path
from trustjev.checkpoint import load_checkpoint
from trustjev.modeling import load_tokenizer
from trustjev.inference import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--base-model", help="Override the saved local backbone directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    model = load_checkpoint(args.checkpoint, device=args.device, base_model=args.base_model)
    result = predict(
        model,
        load_tokenizer(model.config),
        json.loads(Path(args.input).read_text()),
        device=args.device,
        threshold=args.threshold,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
