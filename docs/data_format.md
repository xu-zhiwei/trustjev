# Data format

Provide UTF-8 JSONL files. Each line contains one complete group of related
labeled inputs. Grouping keeps paired views together during training.

Use the same record schema for `train.jsonl`, `dev.jsonl` and `test.jsonl`.
These files divide the data by its role in training, selection and final
evaluation. The general grouped schema supports user-defined finite-candidate
applications. Authentication is an optional pair specialization, and an unlabeled
request is the inference interface for a single input. See the
[file organization guide](../examples/README.md) for the layout and templates.

## Group fields

| Field | Meaning |
| --- | --- |
| `id` | Nonempty group identifier, unique within a dataset and split |
| `dataset` | User-defined source identifier for balancing and reporting |
| `task` | User-defined task identifier for macro evaluation |
| `cluster` | Optional parent identifier for split-overlap checks |
| `views` | Nonempty list of labeled input variants |
| `pairing` | `explicit` by default, or `interventions` |
| `adversarial` | Boolean indicating whether to build an online attack bank for this group |
| `verified_invariance_pairs` | Optional pairs of view names with matching labels |
| `verified_counterfactual_pairs` | Optional pairs of view names with different labels |

Dataset names are supplied by the user. They do not select training behavior.
`pairing` determines paired supervision, and `adversarial` selects groups for
online attack construction.

The loader rejects duplicate IDs and declared train/development group or cluster
overlap. Data preparation supplies checked labels, valid intervention relations
and semantically separate splits.

A sidecar named `<stem>_eligibility.json`, when present next to a JSONL file,
contains an `accepted_ids` list. Without a sidecar, all records are used. Missing
or duplicate accepted IDs raise errors. Inputs exceeding the configured token
limit are rejected rather than truncated.

## View fields

| Field | Meaning |
| --- | --- |
| `name` | Unique view name within the group |
| `policy` | Trusted application-supplied policy text |
| `evidence` | External evidence text |
| `options` | At least two distinct nonempty candidate strings |
| `label` | Zero-based index into `options` |
| `requires_evidence` | Boolean training annotation for authority supervision |
| `attack` | Boolean attack/non-attack annotation for evaluation |

Candidate strings and their order agree across views in a group.
`requires_evidence` supervises training and is not supplied to the deployed model.
An unlabeled prediction request needs just `policy`, `evidence` and `options`.

## Pairing modes

`explicit` groups declare their checked relations using
`verified_invariance_pairs` and `verified_counterfactual_pairs`. The former have
equal labels, and the latter have different labels.

`interventions` groups contain six named views
`clean`, `injected`, `fact_change`, `policy_change`, `trusted_override` and
`untrusted_override`. Injected and untrusted override views receive invariance
supervision when the clean label is unchanged. Fact, policy and trusted-override
views receive preference-change supervision when their labels differ from clean.

Optional operator interventions contain both `operator_clean_or` and
`operator_injected_or`. Their invariant pair is supervised, and the clean operator
view enters preference-change supervision when its label differs from `clean`.
These views are supplied by data preparation.

The objective averages relations separately within each pairing mode and adds
the two mode-level means. It does not use dataset names to choose a mode.

## Online adversarial refinement

Set `adversarial` to `true` on participating training groups. The first view needs
to be named `clean`. For each incorrect candidate, the attack builder appends four
instruction templates and four template compositions, yielding eight variants
per incorrect candidate.

When either adversarial coefficient is nonzero, provide
`--validator module:function`. The callable receives a proposed view dictionary
and returns the correct candidate string under the policy and relevant facts.
It checks each constructed attack without mutating the input.

```python
def label_candidate(view: dict) -> str:
    # Implement the dataset's independent label oracle.
    ...
```

A changed label, a mutated view or the absence of participating groups raises
an error. To train without online attack construction, set both adversarial
coefficients to zero. The package does not supply a dataset-specific rule oracle.

## Authentication

Each group contains a legitimate view and an attack view with the candidates
`Access Granted` and `Access Denied`. The legitimate label selects the former,
and the attack label the latter. Candidate order can differ between groups but
agrees within each pair. Authentication adaptation processes a complete pair
per microbatch and averages gradients over the configured number of groups.
