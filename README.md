# TrustJev

Trust-aware counterfactual learning for finite-candidate decisions.

TrustJev combines protected policy scores with bounded evidence corrections.
Paired supervision encourages stable decisions under fact-preserving instruction
insertions and changed preferences when facts or authorized policies change.

The package provides model construction, fine-tuning, checkpoint loading,
evaluation, authentication threshold calibration and prediction. Datasets and
model weights are supplied by the user.

## Method overview

![TrustJev workflow showing protected policy scoring, bounded evidence corrections, and paired and authority supervision](docs/assets/workflow.png)

*TrustJev workflow from the paper. A shared Transformer produces protected policy
scores and candidate-dependent evidence corrections. Paired and authority
supervision coordinate decision stability and responsiveness to relevant facts.*

## Decision setting

![A policy-based refund workflow with benign requests and an injected instruction in external evidence](docs/assets/decision-example.png)

*Illustrative decision setting from the paper. Customer requests provide facts,
while the application supplies the policy and candidates. An instruction embedded
in a request does not acquire authority to change the policy.*

## Installation

Use Python 3.10 or newer and a CUDA-capable PyTorch installation for training.

```bash
python -m pip install -e .
```

Provide a local model and tokenizer directory with `--base-model`. The backbone
needs to support explicit position IDs, an additive four-dimensional attention
mask and token-level hidden states. Configure the model's adapter target modules
and three existing control tokens in the YAML files. See
[model configuration](docs/training.md#model-configuration) for details.

## Data

Training consumes JSONL files containing complete groups of related labeled
inputs. Prediction needs a trusted policy, external evidence and candidate
strings. The [data interface](docs/data_format.md) describes the fields and
checked pair annotations.

Organize labeled data into `train.jsonl`, `dev.jsonl` and `test.jsonl` using the
same record format. [File organization and input templates](examples/README.md)
provides a general grouped format, an optional grant/deny pair format, and a
separate unlabeled input for single-request inference. The grouped format accepts
user-defined policies, evidence and candidate sets.

Examples below use placeholder paths. No dataset or model files are bundled.
The label validator is supplied as an importable `module:function` callable.

## Fine-tuning

The task-training stages learn the scoring model and refine its evidence path.
Authentication adaptation is an additional stage for grant/deny decisions.
The configuration files expose the architecture and optimization settings.

The supplied training hyperparameters are reference settings used in our
experiments. Adapt learning rates, epochs, group batch sizes, adapter settings
and loss weights to your model, dataset and available resources, and assess the
choices on development data. See [parameter guidance](docs/training.md#reference-training-settings).

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --config configs/stage1.yaml --base-model /path/to/model \
  --train /path/to/dataset/train.jsonl \
  --development /path/to/dataset/dev.jsonl \
  --validator dataset_module:label_candidate --output outputs/stage1

CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --config configs/stage2.yaml --base-model /path/to/model \
  --train /path/to/dataset/train.jsonl \
  --development /path/to/dataset/dev.jsonl \
  --validator dataset_module:label_candidate \
  --warm-start /path/to/stage1/checkpoint --output outputs/stage2

CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --config configs/authentication.yaml --base-model /path/to/model \
  --train /path/to/authentication_dataset/train.jsonl \
  --development /path/to/authentication_dataset/dev.jsonl \
  --warm-start /path/to/stage2/checkpoint --output outputs/authentication
```

Task-training checkpoints are selected using development Balance and recorded in
`selection.json`. Supply the appropriate checkpoint when starting the next stage.
For multiple sources, list their files after `--train` and `--development`.
[Training documentation](docs/training.md) explains the objectives and reductions.

## Evaluation and prediction

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --checkpoint /path/to/checkpoint \
  --data /path/to/dataset/test.jsonl --output outputs/evaluation

CUDA_VISIBLE_DEVICES=0 python scripts/predict.py \
  --checkpoint /path/to/checkpoint --input /path/to/request.json
```

Prediction requests contain `policy`, `evidence` and `options`. They do not need
training labels. Supply `--base-model` if the model directory has moved since the
checkpoint was saved.

Authentication uses a threshold on the grant-minus-deny score difference. Fit it
on development predictions, then keep it fixed during evaluation.

```bash
python scripts/calibrate.py \
  --development-predictions /path/to/development_predictions.json \
  --output outputs/authentication_threshold.json

CUDA_VISIBLE_DEVICES=0 python scripts/predict.py \
  --checkpoint /path/to/authentication/checkpoint \
  --input /path/to/request.json --threshold YOUR_VALIDATION_THRESHOLD
```

## Documentation

- [Method](docs/method.md)
- [Data format](docs/data_format.md)
- [Training](docs/training.md)

## License

This project is licensed under the [MIT License](LICENSE). Third-party models,
datasets and dependencies retain their respective licenses.
