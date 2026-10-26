"""Train a single TrustJev stage with externally supplied labeled groups."""

import argparse
from trustjev.configuration import load_config
from trustjev.training.common import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-model", required=True, help="Local model and tokenizer directory")
    parser.add_argument("--train", nargs="+", required=True, help="Ordered input JSONL files")
    parser.add_argument("--development", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warm-start")
    parser.add_argument("--validator", help="External label oracle as module:function")
    args = parser.parse_args()
    config = load_config(args.config)
    config.model.base_model = args.base_model
    run_training(
        config,
        args.train,
        args.development,
        args.output,
        warm=args.warm_start,
        validator_reference=args.validator,
    )


if __name__ == "__main__":
    main()
