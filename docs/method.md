# TrustJev method

TrustJev scores a finite candidate set using protected policy preferences and
bounded corrections from external evidence. Both scoring branches share a
Transformer backbone.

| Component | Implementation |
| --- | --- |
| Application-owned source boundaries | `encoding.text_tokens`, `encoding.encode` |
| Protected physical prefix | `encoding.protected_prefix` |
| Attention permissions and logical positions | `encoding.batch_inputs`, `encoding.encode` |
| Policy scores and trusted budget | `modeling.ProvenanceDecisionModel.forward` |
| Candidate-dependent evidence representation | Evidence pooling and contextual readout in `forward` |
| Centered bounded correction | `modeling.bounded_residual` |
| Checked-pair supervision | `losses.paired` |
| Authority supervision and gradient routing | `losses.authority` |
| Policy-sufficient margin | `losses.margin` |
| Stochastic consistency | `losses.consistency` |
| Online adversarial refinement | `training.adversarial` |
| Authentication adaptation | `training.authentication`, `losses.authentication` |

## Protected policy computation

The application supplies policy, evidence and candidates with known source
boundaries. Candidate text occurs in an authority branch and a contextual branch.
Policy states read policy tokens. Authority-candidate states read policy and their
own candidate. Evidence states read policy and evidence. Contextual candidates
additionally read evidence. Different candidate branches do not read one another.

Authority candidate positions begin immediately after the policy, independent
of evidence length. Contextual candidate positions begin after policy and
evidence. Source-marker spellings embedded in supplied text are escaped.

Projected policy and authority-candidate representations produce

[
a_k=A([p,u_k,podot u_k]),
qquad
eta=(B+	frac12log K)sigma(G([p,ar u])).
]

The configured ceiling parameter is (B). The shared budget depends on policy
and candidates rather than evidence.

## Bounded evidence correction

The refined evidence representation mixes candidate-query evidence pooling
and contextual candidate states. A trainable policy–evidence product projection
enters the evidence head before its nonlinearity.

For valid candidates, the scorer computes

[
z_k=a_k+eta	anh(e_k-ar e).
]

Padding candidates are excluded from the mean. Each correction is bounded by
(eta). Centered evidence scores sum to zero, while the transformed corrections
need not. If an authority margin exceeds (2eta), the authority-preferred
candidate is stable to evidence changes under the isolated computation and exact
arithmetic. This condition does not establish policy correctness or constitute
a finite-precision proof. The `authority_stable` prediction field concerns the
argmax authority preference, not a thresholded authentication decision.

## Supervision

Endpoint cross-entropy supervises each labeled view. Checked invariant pairs
use symmetric KL between candidate distributions. Different-label pairs use the
change in relative preference between their two correct candidates.

Authority supervision favors the correct policy candidate when policy suffices
and a uniform authority distribution when external evidence is needed. The
policy-sufficient margin target includes the correction bound. Refinement routes
the uniformity term to the authority head while retaining other shared gradients.

Online refinement checks candidate attacks with a supplied label oracle,
selects the highest-loss attack and backpropagates adversarial classification and
consistency losses. Authentication uses paired grant/deny ranking alongside
classification and stochastic prediction consistency.

[Training](training.md) describes the configurable stages and reductions.
