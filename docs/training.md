# Training

The package provides two task-training stages and optional authentication
adaptation. Each stage starts a new optimizer and learning-rate schedule.

## Reference training settings

The numeric training hyperparameters in the example configurations and tables
are settings used in our experiments, provided as starting points for users.
Their suitability for another model or dataset needs to be assessed with
representative development data.

Adjust learning rates, epoch counts, group batch sizes, weight decay, adapter
rank and dropout, and configurable loss weights according to the model,
data characteristics and available compute. Use the development split for
parameter selection and keep the test split for final evaluation.

Stage architecture settings such as the protected prefix, product interaction,
evidence mixture and centering are validated against the stage definitions.
Keep those settings consistent with the chosen stage when adjusting training
hyperparameters.

## Model configuration

`model.base_model` identifies the local model and tokenizer directory. The
backbone should return token-level `last_hidden_state` values and accept
`position_ids`, an additive four-dimensional `attention_mask` and `use_cache`.
Its attention implementation needs to respect the supplied mask at every layer.
The model loads in bfloat16, while trainable readouts compute in float32.

`model.target_modules` lists the backbone projection names to adapt with LoRA.
Adjust it to the selected model's module names. `model.control_tokens` provides
three distinct existing special-token spellings for policy start, evidence or
candidate start, and segment end. The tokenizer needs to map them to distinct
valid token IDs. The example spellings and projection names in the configurations
are interface defaults, not a guarantee that every backbone supports them.
Backbones with different attention interfaces need an adapter before use.

The other model settings control adapter rank, dropout, evidence budget,
evidence mixture, protected prefix, product interaction and sequence length.
The sequence limit includes the policy, evidence, candidate copies and markers.

## Training stages and reference settings

| Setting | Initial task training | Task refinement | Authentication |
| --- | --- | --- | --- |
| Configuration | `stage1.yaml` | `stage2.yaml` | `authentication.yaml` |
| Epochs | 3 | 2 | 2 |
| Learning rate | 0.0001 | 0.00005 | 0.0001 |
| Effective group batch | 2 | 2 | 4 |
| Contextual evidence mixture | 1 | 0.5 | 0.5 |
| Product interaction | Absent | Zero-initialized | Retained |
| Authority candidates before evidence | No | Yes | Yes |
| Uniformity gradients | Shared representations and head | Authority head | Not applied |

Initial task training uses the contextual candidate evidence representation.
Refinement combines explicit evidence pooling with contextual candidate states
and introduces a zero-initialized policy–evidence product projection.
Both stages isolate authority attention and use logical candidate positions.
Authentication retains the refined architecture and adapts it to paired
legitimate and attack requests.

## Objectives and reference weights

| Term | Initial task training | Task refinement | Authentication |
| --- | --- | --- | --- |
| Endpoint cross-entropy | 1 | Two-pass mean, weight 1 | Two-pass mean, weight 1 |
| Stochastic symmetric KL | Not applied | 1 | 1 |
| Checked input invariance | 0.03 | 0.0075 | Not applied |
| Checked preference change | 0.03 | 0.0075 | Not applied |
| Authority supervision | 0.1 | 0.025 | Not applied |
| Policy-sufficient margin | 0.1 | 0.025 | Not applied |
| Adversarial cross-entropy | 1 | 0.25 | Not applied |
| Adversarial consistency | 0.03 | 0.0075 | Not applied |
| Authentication pair ranking | Not applied | Not applied | 1 |

The YAML files expose these coefficients. Margin slack and authentication ranking
margin are 1. Task refinement applies paired, authority and margin supervision
to the first prediction pass. Its two passes use separate dropout draws.

Endpoint cross-entropy, margin and stochastic consistency use complete-group
means weighted by inverse source sampling probability. Initial authority targets
average policy-sufficient and evidence-required views separately across the
minibatch. Refinement forms those terms within each group and source-balances
the group contributions. Missing strata contribute zero.

Paired supervision keeps separate relation-count means for explicit pairs and
named interventions. Adversarial losses account for the participating source's
sampling probability without renormalizing to the participating groups.

In refinement, the evidence-required uniformity term updates the authority head
without contributing gradients to shared representations. Other objective
gradients remain active. Authentication averages complete-pair gradients over
the configured group batch.

## Optimization and checkpoints

AdamW uses the configured learning rate and weight decay. Gradients are clipped
to norm 1. The schedule uses a linear warm-up followed by linear decay with a
minimum multiplier of 0.1. Each epoch shuffles complete groups. Gradient
checkpointing uses the non-reentrant implementation, and TF32 matrix
multiplication is disabled.

Epoch checkpoints contain `adapter/`, `head.pt`, `model_config.json` and
`metadata.json`. Warm starts validate the predecessor stage and structural
settings. Promoting the initial model adds its missing product projection at
zero. Other readout and adapter parameters are required.

Task-training selection maximizes development Balance per dataset and keeps the
earliest epoch on ties. Authentication selection retains the last configured
epoch. The caller passes the selected checkpoint explicitly to the next stage.
Use identical model and tokenizer assets across stages.

Outputs include training configuration, input hashes, package versions,
development predictions and selection records. Existing run directories are
not overwritten. Checkpoints save model weights and metadata, rather than
optimizer and random-generator resume state.

## Authentication threshold

Return `Access Granted` when the grant-minus-deny score exceeds the supplied
threshold. Fit the threshold from development predictions with `calibrate.py`
and hold it fixed during evaluation.

The calibration command accepts one or more prediction files for the same
labeled development views. It selects a common threshold by mean Balance, then
minimum model Balance, then smallest absolute threshold. Calibration output
reports the fitted development metrics and threshold.
