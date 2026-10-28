"""Actual finite-field/CRT behavior, including coordinated modifications."""
import gmpy2
import numpy as np
from experiments.point1_compare.adapters.finite_field import FiniteFieldHH, FFDHE3072_P, FFDHE3072_Q
from experiments.point1_compare.adapters.saev_full import SAEVProtocol


def test_finite_field_parameters_and_negative_homomorphism():
    assert FFDHE3072_P.bit_length() == 3072
    assert gmpy2.is_prime(FFDHE3072_P) and gmpy2.is_prime(FFDHE3072_Q)
    hh = FiniteFieldHH(3)
    assert all(pow(g, hh.q, hh.p) == 1 and g != 1 for g in hh.generators)
    x, y = np.array([5, -7, 0]), np.array([-10, 3, 2])
    assert hh.hash(x+y) == hh.hash(x)*hh.hash(y) % hh.p


def test_saev_real_crt_honest_dropout_and_coordinated_proof():
    vectors = {f"u{i}": np.array([i-3, i*7, -i, 0], dtype=np.int64) for i in range(4)}
    state = SAEVProtocol(4, [["u0", "u1"], ["u2", "u3"]])
    honest = state.run_round(vectors, set(vectors), 0)
    assert honest.verified and honest.extra["all_recovered_vectors_equal"]
    np.testing.assert_array_equal(honest.raw_sum, sum(vectors.values()))
    online = set(vectors)-{"u0"}
    dropout = state.run_round(vectors, online, 1)
    assert dropout.verified and dropout.extra["all_recovered_vectors_equal"]
    np.testing.assert_array_equal(dropout.raw_sum, sum(vectors[u] for u in sorted(online)))
    uncoordinated = state.run_round(vectors, set(vectors), 2, attack="result_only")
    assert not uncoordinated.verified
    forged = state.run_round(vectors, set(vectors), 3, attack="coordinated")
    expected = sum(vectors.values())
    expected[0] += 1
    assert forged.verified and forged.extra["all_recovered_vectors_equal"]
    np.testing.assert_array_equal(forged.raw_sum, expected)
