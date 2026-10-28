"""Bounded literal-equation diagnostics, NOT full baseline reproductions.

SAEV uses a tiny legal group to check algebra only. OVP uses real BLS12-381
target-group operations. Neither supplies full privacy/dropout/performance
results. VLMRS tests equations 4-10 with an explicitly instantiated keyed
linear HF. None by itself establishes a vulnerability in a complete paper.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import secrets
import sys
from pathlib import Path
from .artifacts import create_run, digest, dump_json
from .run_compare import ROOT, _git_commit, _lib_versions, _new_run_id, _review_metadata


def saev_equations():
    # Prime-order subgroup of F_p*, not an EC point encoded as an integer.
    # Small parameters are deliberately unsuitable for security/performance claims.
    p, q, generator = 467, 233, 4
    assert pow(generator, q, p) == 1 and generator != 1
    moduli = [239, 241, 251, 257]
    values = [3, 5, 7, 9]
    assert min(moduli) > len(values)*max(values)
    product = math.prod(moduli)
    coefficients = [(product//a)*pow(product//a, -1, a) for a in moduli]
    masks = [secrets.randbelow(p) for _ in values]
    blinders = []
    while len(blinders) < len(values):
        value = pow(generator, secrets.randbelow(q-1)+1, p)
        if value not in blinders:
            blinders.append(value)
    ciphertext = sum(x+c*noise for x, c, noise in zip(values, coefficients, masks))
    proof = math.prod(r*pow(generator, x, p) % p for r, x in zip(blinders, values)) % p
    k, unblind = sum(coefficients), math.prod(blinders) % p

    def verify(candidate, candidate_proof):
        recovered = [(k*candidate-c*c*noise) % a
                     for c, noise, a in zip(coefficients, masks, moduli)]
        accepted = [pow(generator, total, p) == candidate_proof*pow(unblind, -1, p) % p
                    for total in recovered]
        return recovered, accepted

    honest_sum, honest = verify(ciphertext, proof)
    delta = 7
    # Cloud construction uses only its ciphertext/proof, delta and public HH.
    fake_ciphertext = ciphertext+delta
    fake_proof = proof*pow(generator, delta, p) % p
    fake_sum, forged = verify(fake_ciphertext, fake_proof)
    _, uncoordinated = verify(fake_ciphertext, proof)
    assert honest_sum == [sum(values)]*len(values) and all(honest)
    assert fake_sum == [sum(values)+delta]*len(values)
    return {"scope": "Scalar CRT/decryption and multiplicative HH equations only; toy group",
            "security_bits": None, "full_protocol": False, "honest_accepted": all(honest),
            "result_only_tamper_accepted": any(uncoordinated),
            "coordinated_ciphertext_and_proof_accepted": all(forged),
            "attacker_uses_ta_secrets": False, "p": p, "q": q,
            "honest_aggregate": honest_sum[0], "tampered_aggregate": fake_sum[0]}


def ovp_equations():
    from py_ecc.optimized_bls12_381 import G1, G2, pairing, multiply, curve_order
    q = curve_order
    base = pairing(G2, G1)
    label = b"ovp-literal-equation-probe|round=1"
    hashed_round = int.from_bytes(hashlib.sha256(label).digest(), "big") % q
    public_authentication = base**hashed_round
    # These are post-unmasking sums at the two verification servers. The
    # diagnostic starts here; DH/AES, individual uploads and dropouts are absent.
    partial = [secrets.randbelow(q), secrets.randbelow(q)]
    betas = [secrets.randbelow(q-1)+1, secrets.randbelow(q-1)+1]
    actual = sum(partial) % q
    tags = [base**((hashed_round+beta)*part % q) for beta, part in zip(betas, partial)]
    mac = tags[0]*tags[1]
    assert pairing(multiply(G2, (hashed_round+betas[0]) % q),
                   multiply(G1, partial[0])) == tags[0]

    def honest_services_and_user(candidate, candidate_mac, opposite):
        # The services use their own beta. No beta is passed to the cloud.
        service_mac = [base**(beta*other % q) for beta, other in zip(betas, opposite)]
        service_small = [base**(beta*candidate % q) for beta in betas]
        left = candidate_mac*service_mac[0]*service_mac[1]
        right = public_authentication**candidate*service_small[0]*service_small[1]
        local_sum_checks = [(own+other) % q == candidate for own, other in zip(partial, opposite)]
        return left == right, local_sum_checks

    opposite = [partial[1], partial[0]]
    honest, _ = honest_services_and_user(actual, mac, opposite)

    def cloud_forge(candidate, candidate_mac, cross_sums, public_round_auth, delta):
        return ((candidate+delta) % q, candidate_mac*(public_round_auth**delta),
                [(s+delta) % q for s in cross_sums])

    fake, fake_mac, fake_opposite = cloud_forge(actual, mac, opposite, public_authentication, 7)
    forged, local_sum_checks = honest_services_and_user(fake, fake_mac, fake_opposite)
    uncoordinated, _ = honest_services_and_user(fake, mac, opposite)
    assert honest and fake != actual
    return {"scope": "Scalar post-unmasking feedback/verification equations with real pairings",
            "curve": "BLS12-381", "full_protocol": False, "honest_accepted": honest,
            "result_only_tamper_accepted": uncoordinated,
            "coordinated_result_mac_and_feedback_accepted": forged,
            "both_services_own_plus_claimed_opposite_equals_claimed_total": local_sum_checks,
            "attacker_uses_service_betas": False,
            "condition": "Public hash(round), public g/h, literal feedback equations; no additional authenticated cross-sum binding",
            "honest_aggregate": actual, "tampered_aggregate": fake}


def vlmrs_equations():
    import numpy as np
    from .adapters.vlmrs_verification import VLMRSVerification, coordinated_scale
    users = ["u0", "u1", "u2"]
    inputs = [np.array([i, -i-2, 0, 17-i, -31, 8, 2*i, 1], dtype=np.int64)
              for i in range(len(users))]
    verifier = VLMRSVerification(8, root_exponent=7)
    proof = verifier.aggregate(verifier.create(user, vector) for user, vector in zip(users, inputs))
    actual = np.sum(inputs, axis=0, dtype=np.int64)
    honest, honest_checks = verifier.verify(actual, proof, users)
    fake, fake_proof = coordinated_scale(actual, proof, 2, 7)
    uncoordinated, _ = verifier.verify(fake, proof, users)
    forged, forged_checks = verifier.verify(fake, fake_proof, users)
    assert honest and not uncoordinated and not np.array_equal(fake, actual)
    return {"scope": "Equations 4-10 only; keyed linear vector HF is an explicit implementation choice",
            "curve": "BLS12-381", "full_protocol": False,
            "condition": "Root exponent d is public and invertible modulo the prime group order; no additional authenticated binding of proof to uploads",
            "honest_accepted": honest, "honest_checks": honest_checks,
            "result_only_tamper_accepted": uncoordinated,
            "coordinated_scaling_accepted": forged, "coordinated_checks": forged_checks,
            "attacker_uses_hf_or_prf_secrets": False,
            "honest_aggregate": actual.tolist(), "tampered_aggregate": fake.tolist()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairing-deps", default=str(Path(ROOT)/"tmp"/"point1_pairing_deps"))
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path(args.pairing_deps).resolve()))
    import py_ecc
    import py_ecc.optimized_bls12_381  # Load the actually used dependencies before archiving source.
    run_id = _new_run_id("baseline_equation_probe")
    manifest = {"run_id": run_id, "experiment": "algebra_probe", "command": " ".join(sys.argv),
                "seed": None, "schemes": ["saev_equation_probe", "ovp_equation_probe", "vlmrs_equation_probe"],
                "git_commit": _git_commit(), "lib_versions": {**_lib_versions(), "py_ecc": py_ecc.__version__},
                "quant": None, "protocol_mapping": "docs/POINT1_BASELINE_IMPLEMENTATION_REVIEW_20261018.md",
                "paper_versions": {"saev": "10.1109/TIV.2025.3599909", "ovp": "10.1109/TNSE.2025.3543601",
                                   "vlmrs": "10.1109/TDSC.2025.3606413"},
                "concurrency": "serial", "timing_scope": "no performance observation", "verification": "equations only",
                "aggregation": "scalar diagnostics", "execution_status": "running", "full_protocol": False}
    manifest.update(_review_metadata(vars(args)))
    directory = Path(ROOT)/"results"/"point1_compare"/run_id
    create_run(directory, manifest, Path(ROOT))
    result = {"saev_literal_equations": saev_equations(), "ovp_literal_equations": ovp_equations(),
              "vlmrs_literal_equations": vlmrs_equations()}
    dump_json(directory/"probe.json", result)
    manifest["output_sha256"] = {"probe.json": digest(directory/"probe.json")}
    manifest["execution_status"] = "completed"
    dump_json(directory/"manifest.json", manifest)
    print(directory)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
