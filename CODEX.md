# Dual-TA HVSA Notes

## Goal

This repository implements a hierarchical federated learning scheme with:

- one `CloudServer`
- multiple `EdgeServer`s
- many `User`s
- two logical trusted authorities: `TA1` and `TA2`

The design target is:

- keep user-side cost near `O(d)`
- avoid a single fully trusted TA
- preserve dropout tolerance through online-user lists
- keep verification at the edge layer instead of burdening users

## Current Code Anchors

- `user.py`: user registration, local training, dual-TA masking, upload
- `ta.py`: `TrustedAuthority`, per-user seed agreement, partial mask computation, mask hash generation
- `edge.py`: edge aggregation, online-user reporting, signature generation, global-result verification
- `cloud.py`: masked-sum collection, partial-mask collection, final unmasking and averaging
- `utils.py`: key agreement, PRG, LHH, signatures, quantization helpers

## Core Architecture

### 1. Registration

Each user establishes one secret seed with `TA1` and one secret seed with `TA2`.

For user `u_i` at round `t`:

- `m_i^(1)(t) = PRG(s_i^(1), u_i, t)`
- `m_i^(2)(t) = PRG(s_i^(2), u_i, t)`
- `m_i(t) = m_i^(1)(t) + m_i^(2)(t)`

The user uploads only one masked vector:

- `wbar_i(t) = w_i(t) + m_i(t)`

Do not split the user upload into two shares unless the whole protocol is intentionally redesigned.

### 2. Edge Aggregation

Each edge server:

- collects masked uploads from online users
- computes the local masked sum
- optionally computes `LHH` and signs the local evidence
- sends the online user list to both `TA1` and `TA2`
- forwards the masked aggregate to the cloud

### 3. Dual-TA Partial Unmasking

`TA1` and `TA2` independently compute partial global masks from the same online-user set:

- `M_global^(1)(t) = sum m_i^(1)(t)`
- `M_global^(2)(t) = sum m_i^(2)(t)`

The cloud must collect both partial masks before unmasking:

- `W_sum(t) = sum_k Wbar_k(t) - M_global^(1)(t) - M_global^(2)(t)`

This is a `2-of-2` logical protocol, not a threshold TA design.

### 4. Verification

Verification should remain edge-centric.

Recommended relation:

- `H_sum(t) * H_mask^(1)(t) * H_mask^(2)(t) = product_k H_k(t)`

Where:

- `H_k(t)` is the LHH hash of edge `k`'s masked aggregate
- `H_sum(t)` is the LHH hash of the recovered global plaintext sum
- `H_mask^(1)(t)` and `H_mask^(2)(t)` are the hashes of the two partial global masks

Important:

- the two TAs provide mask evidence
- the edge servers remain the main verification executors
- do not move full verification logic into both TAs unless there is a strong research reason

## Threat Model

The intended trust assumptions are:

- `TA1` and `TA2` are independent logical authorities
- each TA is semi-honest
- `TA1` and `TA2` do not collude
- the cloud may be malicious
- edge servers are honest-but-curious unless a specific experiment changes that assumption

Security intuition:

- `CS + TA1` is insufficient to recover honest user updates
- `CS + TA2` is insufficient to recover honest user updates
- privacy fails only if both TAs collude or both sides leak the necessary secrets

## Review Boundary: Collusion and Secure Channels

The following assumptions are fixed review premises for the single-TA research-point and thesis documents. They do not replace the separate dual-logical-TA implementation invariants below:

1. Do not assess anti-collusion security by replacing the intended multi-party aggregation with an artificial two-participant scenario. For an actually released aggregate containing `N` clients, the supported maximum-collusion model allows the Cloud to collude with all ESs and with at most `c <= N - 2` clients, so at least two non-colluding clients remain: `N - c >= 2`. In Research Point 1, `N = N_online(t)`; in Research Point 2, `N = N_Q(t) = sum_{Pi_k in Q} n_k(t_k)`. `n_min >= 2` is only a package/round admission guard and is not itself the collusion threshold. A timeout checkpoint containing one group is valid when that group still satisfies the client-level threshold; do not replace the client threshold with `|Q| >= 2`.
2. Protocol communication uses ideal secure channels. Messages are not distorted, substituted, forged, or misattributed in transit. A message that an ES receives from a fully trusted TA is assumed authentic and correct by definition.
3. The Cloud/CS is untrusted. Values claimed or produced by the Cloud must still be verified wherever the protocol specifies verification. The secure-channel assumption authenticates the sender and preserves the message; it does not make Cloud-generated content trustworthy.
4. Consequently, do not report missing transport authentication, TA-message signatures, or TA-to-ES provenance plumbing in the experimental code as cryptographic design flaws when the code is abstracting the ideal secure channel. Review whether Cloud-produced results are checked against authentic TA/ES evidence instead.
5. TA is fully trusted and never colludes. The Cloud may know the aggregate mask because it must unmask the selected aggregate; secrecy of the aggregate mask is not a security goal. The privacy goal protects each individual client's parameters, while ES/group aggregates and Cloud/global aggregates may be disclosed.
6. Global models, historical model versions, and protocol-declared public hyperparameters are consensus values. Do not turn cache loss, stale local copies, or transmission failures for these values into paper-level security flaws under the ideal-channel model.

## Availability Assumption

This repository should treat the protocol as:

- dual logical TAs at the protocol layer
- primary/standby deployment at the system layer

That means:

- protocol security comes from dual authorities
- availability comes from engineering redundancy, not threshold cryptography

Do not silently upgrade the design to threshold TA unless the paper chapter, threat model, and experiments are updated together.

## Implementation Rules

When modifying the dual-TA pipeline, preserve these invariants:

1. Users must negotiate independent seeds with both TAs.
2. Users should upload one masked vector, not two separate protocol branches.
3. Both TAs must derive masks only from registered seeds plus round metadata.
4. Online-user lists must be the source of dropout recovery.
5. The cloud must not unmask until all required partial masks arrive.
6. Verification should use edge-produced evidence plus TA mask hashes.
7. The encoding path and aggregate-domain arithmetic must stay consistent across `user.py` and `cloud.py`.

## Anti-Patterns

Avoid these changes unless the whole protocol is intentionally reworked:

- turning one TA into a full PKI authority and the other into a dummy helper
- letting only one TA determine the final online-user set
- allowing the cloud to continue after receiving only one partial mask
- mixing single-TA and dual-TA formulas in the same round path
- coupling verification correctness to user-side heavy crypto
- claiming threshold fault tolerance without implementing real threshold recovery

## Recommended Chapter Framing

If this design is written into a thesis chapter, frame it as:

"A lightweight verifiable secure aggregation scheme with dual assisted authorities."

Recommended emphasis:

- reduced single-point trust compared with a single-TA design
- preserved user-side `O(d)` cost
- dropout tolerance via online-user reporting
- edge-offloaded verification
- system-layer high availability through primary/standby deployment

## If You Extend This Design

Safe extensions:

- add clearer TA1/TA2 naming throughout logs and experiment scripts
- add explicit verification of `H_mask^(1)` and `H_mask^(2)`
- add experiments comparing single-TA and dual-TA overhead
- add primary/standby failover simulation for each logical TA

Heavy extensions that require a full redesign:

- threshold TA
- user-to-user secret sharing recovery
- asynchronous mask release
- moving verification authority from edges to TAs

## Short Summary

This codebase should be treated as a dual-TA, edge-verified, hierarchical secure aggregation system.

Keep the split clean:

- users mask once
- edges aggregate and verify
- both TAs provide partial unmasking material
- the cloud combines, unmasks, and averages only after both TA contributions arrive
