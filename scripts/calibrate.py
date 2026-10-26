"""Fit a deployment threshold from development predictions."""

import argparse
import json
from pathlib import Path
from trustjev.calibration import calibrate
from trustjev.checkpoint import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-predictions", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    result = calibrate([json.loads(Path(p).read_text()) for p in args.development_predictions])
    write_json(args.output, result)


if __name__ == "__main__":
    main()
