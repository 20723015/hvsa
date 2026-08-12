# Single-TA HVSA Notes

## Goal

This repository implements a hierarchical federated learning scheme with:

- one `CloudServer` (untrusted, may be malicious)
- multiple `EdgeServer`s (verify Cloud results, sign group evidence)
- many `User`s
- one trusted authority (`TA`)

The design target is:

- keep user-side cost near `O(d)`
- single-TA lightweight verifiable secure aggregation (research points 1 and 2)
- preserve dropout tolerance through online-user lists
- keep verification at the edge layer instead of burdening users

## Current Code Anchors

- `user.py`: user registration, local training, single-TA masking, upload
- `ta.py`: `TrustedAuthority`, per-user seed agreement, global-mask computation, mask-hash generation
- `edge.py`: edge aggregation, online-user reporting, metadata signature generation, global-result verification, point-2 buffer/checkpoint
- `cloud.py`: masked-sum collection, unmasking to `W_sum^raw` (does NOT produce the canonical model)
- `utils.py`: key agreement, PRG, LHH (hash-to-curve generators), signatures, quantization helpers

## Core Architecture

### 1. Registration

Each user establishes one secret seed with the TA (X25519 + HKDF).

For user `u_i` at round `t`:

- `m_i(t) = PRG(s_i, u_i, t)`

The user uploads only one masked vector:

- `wbar_i(t) = Quantize(w_i(t)) + m_i(t)`

### 2. Edge Aggregation

Each edge server:

- collects masked uploads from online users
- computes the local masked sum `Wbar_k(t)`
- computes `H_k(t) = LHH.Hash(pp_LHH, Wbar_k(t))`
- signs the full metadata `meta_k = (edge_id, round, base_hash, digest(U_online), n_k, S_k, H_k)`
- sends the online user list to the TA
- forwards the masked aggregate and signed evidence to the cloud

### 3. TA Global Mask and Cloud Unmask

The TA derives the global mask from the union of online-user lists:

- `M_global(t) = sum_{u in U_online(t)} m_u(t)`
- `H_mask(t) = LHH.Hash(M_global(t))`

The TA sends `M_global(t)` to the cloud (the aggregate mask is not a secret; it must be known to unmask). The cloud computes the verified plaintext sum only:

- `W_sum^raw(t) = sum_k Wbar_k(t) - M_global(t)`

The cloud does NOT generate a canonical global model. The consensus model is reconstructed deterministically by the edge servers after verification:

- `W(t+1) = W_sum^raw(t) / |U_online(t)|`

### 4. Verification

Verification is edge-centric. Each edge checks:

- Ed25519 signatures over the full `meta_k`
- the LHH relation `H_sum^raw(t) * H_mask(t) = product_k H_k(t)`

On success, the edge reconstructs `W(t+1) = W_sum^raw / n_online` as the next consensus model.

## Threat Model

The intended trust assumptions are:

- the TA is fully trusted and never colludes; it holds all user mask seeds
- the cloud is malicious (may tamper, drop, or replay)
- edge servers may collude with the cloud (share the protocol view); privacy is judged per client
- privacy goal: protect each individual client's parameters; group/global aggregates and the aggregate mask may be disclosed

For a released aggregate containing `N` clients with `c <= N - 2` colluding clients, at least two non-colluding clients remain (`N - c >= 2`). In research point 1, `N = N_online(t)`; in research point 2, `N = N_Q(t) = sum_{Pi_k in Q} n_k(t_k)`. `n_min >= 2` is an admission guard, not the collusion threshold.

Security intuition:

- the cloud cannot recover any individual client's parameters from `M_global` and `W_sum^raw` while `N - c >= 2`
- the cloud cannot tamper `W_sum^raw` without being caught (LHH collision resistance) and cannot forge edge evidence (Ed25519)

## Review Boundary: Collusion and Secure Channels

The following assumptions are fixed review premises for the single-TA research-point and thesis documents:

1. Do not assess anti-collusion security by replacing the intended multi-party aggregation with an artificial two-participant scenario. For an actually released aggregate containing `N` clients, the supported maximum-collusion model allows the Cloud to collude with all ESs and with at most `c <= N - 2` clients, so at least two non-colluding clients remain: `N - c >= 2`. In Research Point 1, `N = N_online(t)`; in Research Point 2, `N = N_Q(t) = sum_{Pi_k in Q} n_k(t_k)`. `n_min >= 2` is only a package/round admission guard and is not itself the collusion threshold. A timeout checkpoint containing one group is valid when that group still satisfies the client-level threshold; do not replace the client threshold with `|Q| >= 2`.
2. Protocol communication uses ideal secure channels. Messages are not distorted, substituted, forged, or misattributed in transit. A message that an ES receives from a fully trusted TA is assumed authentic and correct by definition.
3. The Cloud/CS is untrusted. Values claimed or produced by the Cloud must still be verified wherever the protocol specifies verification. The secure-channel assumption authenticates the sender and preserves the message; it does not make Cloud-generated content trustworthy.
4. Consequently, do not report missing transport authentication, TA-message signatures, or TA-to-ES provenance plumbing in the experimental code as cryptographic design flaws when the code is abstracting the ideal secure channel. Review whether Cloud-produced results are checked against authentic TA/ES evidence instead.
5. TA is fully trusted and never colludes. The Cloud may know the aggregate mask because it must unmask the selected aggregate; secrecy of the aggregate mask is not a security goal. The privacy goal protects each individual client's parameters, while ES/group aggregates and Cloud/global aggregates may be disclosed.
6. Global models, historical model versions, and protocol-declared public hyperparameters are consensus values. Do not turn cache loss, stale local copies, or transmission failures for these values into paper-level security flaws under the ideal-channel model.

## Availability Assumption

Availability is an engineering concern, not a cryptographic one in this design. The single TA is assumed available; availability engineering (e.g., primary/standby deployment of the TA) is a system-layer concern and is out of scope for the protocol security model. Do not silently upgrade the design to threshold TA unless the paper chapter, threat model, and experiments are updated together.

## Implementation Rules

When modifying the single-TA pipeline, preserve these invariants:

1. Users negotiate one independent seed with the TA.
2. Users upload one masked vector, not multiple protocol branches.
3. The TA derives masks only from registered seeds plus round/package metadata.
4. Online-user lists must be the source of dropout recovery.
5. The cloud must not unmask until the TA mask arrives; the cloud must not become the source of truth for the consensus model.
6. Verification should use edge-produced evidence plus the TA mask hash; the edge (not the cloud) derives the consensus model after verification.
7. The encoding path and aggregate-domain arithmetic must stay consistent across `user.py` and `cloud.py`.
8. Point-2 exactly-once: each `package_token` is signed/bound by its edge; the append-only `Aggregated_IDs` rejects replay.

## Anti-Patterns

Avoid these changes unless the whole protocol is intentionally reworked:

- making the cloud the source of truth for the global model
- skipping edge verification of the plaintext sum
- moving full verification logic into the TA
- mixing single-TA and dual-TA formulas in the same round path
- coupling verification correctness to user-side heavy crypto
- claiming threshold fault tolerance without implementing real threshold recovery

## Recommended Chapter Framing

If this design is written into a thesis chapter, frame it as:

"A lightweight verifiable secure aggregation scheme under a single trusted authority."

Recommended emphasis:

- single-TA lightweight architecture (research point 1) and its checkpointed semi-async extension (research point 2)
- preserved user-side `O(d)` cost
- dropout tolerance via online-user reporting
- edge-offloaded verification with `O(1)` verification communication (independent of `d`)
- cloud is untrusted: it provides evidence to be verified, never the canonical model

## If You Extend This Design

Safe extensions:

- hash-to-curve generator derivation is already in place; do not revert to deriving generators by publishing a scalar (that leaks the discrete log and breaks LHH collision resistance)
- add experiments comparing verification on/off overhead
- add primary/standby availability simulation for the TA

Heavy extensions that require a full redesign:

- threshold TA
- user-to-user secret sharing recovery
- moving verification authority from edges to TAs
- going back to a dual-TA mask split

## Short Summary

This codebase should be treated as a single-TA, edge-verified, hierarchical secure aggregation system.

Keep the split clean:

- users mask once with the TA seed
- edges aggregate, sign metadata, and verify
- the TA provides the aggregate unmasking material
- the cloud combines and recovers the plaintext sum only
- edges derive and hold the consensus model
