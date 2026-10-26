# Organizing data and prediction inputs

Start by splitting your labeled data into training, development and test files.
Choose the record format for your application, then use that format consistently
across the three splits. The templates below describe input representations.

## Data splits

```text
dataset/
├── train.jsonl       # Update model parameters
├── dev.jsonl         # Select checkpoints and decision thresholds
└── test.jsonl        # Evaluate with the selected checkpoint and fixed threshold
```

These are data partitions. The training stages in `configs/` describe how model
parameters are updated. Each stage uses the training and development files
appropriate to that stage.

File names are configurable. Pass the training and development files to
`train.py` using `--train` and `--development`, and pass the test file to
`evaluate.py` using `--data`.

## Labeled data formats

| Application | Template | Use for |
| --- | --- | --- |
| Finite-candidate decisions with a policy and external evidence | [task.jsonl.template](task.jsonl.template) | `train.jsonl`, `dev.jsonl` and `test.jsonl` |
| Grant/deny decisions using the optional authentication adaptation | [authentication.jsonl.template](authentication.jsonl.template) | `train.jsonl`, `dev.jsonl` and `test.jsonl` |

Use the general grouped format for your own policies, evidence and candidate
sets. The authentication template illustrates the additional pair constraints
of the provided grant/deny adaptation.

Replace placeholder text, IDs, labels and annotations with checked data. Each
line is one complete JSON object representing a source group, with related
views inside `views`. JSONL files contain one group per line, with no outer list
or commas between lines.

### General grouped format

The example in `task.jsonl.template` contains three related views of one source
input. `clean` and `injected` share a label and form an invariance pair.
`fact_change` has a different label and forms a preference-change pair with
`clean`. These views describe supervision relationships within a record.

Keep candidate strings and their order identical within each group. `label` is
a zero-based candidate index. The example labels and evidence-requirement flags
illustrate the format and need values verified for your application. Keep a
source group and its related views in the same split. Use distinct group IDs and
parent clusters across splits.

`adversarial: true` marks training groups for online attack construction and
requires an independent oracle passed through `--validator`. For a recipe without
online attacks, set both adversarial coefficients to zero in the configuration.
Attack banks are constructed from training groups.

### Optional authentication pair format

The authentication template contains a legitimate request labeled
`Access Granted` and an attack request labeled `Access Denied`. Replace the policy
and request placeholders with checked pairs. This format uses the same split
layout above and is needed when using the authentication adaptation.

## Single-request inference

[request.json.template](request.json.template) is the unlabeled version of a
prediction input. Use it with any trained finite-candidate scorer through
`predict.py`. It contains `policy`, `evidence` and `options`, without training
labels or pair annotations. An ordinary JSON object can span multiple lines.

This file is supplied as an individual prediction request, separately from the
labeled data splits.

## External file layout

Keep user-supplied data and model assets outside the source directory. For
example, a workspace can be organized as follows.

```text
workspace/
├── trustjev/                   # Source checkout
├── model/                      # Model and tokenizer assets
├── dataset/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
├── requests/
│   └── request.json
└── outputs/                    # Checkpoints and generated outputs
```

For an application using authentication adaptation, organize its labeled pairs
in their own directory with the same `train.jsonl`, `dev.jsonl` and `test.jsonl`
layout. All directory and file names above are placeholders.

## Optional eligibility files

Place `train_eligibility.json` beside `train.jsonl` to select a subset by ID.
The loader recognizes the corresponding sidecar for other JSONL filenames.
Without a sidecar, it reads all records.

```json
{
  "accepted_ids": ["<group_id>"]
}
```

IDs refer to groups in the adjacent JSONL file. The complete field definitions
are in [the data format documentation](../docs/data_format.md).
