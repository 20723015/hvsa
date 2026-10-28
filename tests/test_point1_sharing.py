from itertools import combinations
import pytest
from experiments.point1_compare.adapters.sharing import share_secret, reconstruct_secret, PRIME


def test_real_shamir_reconstructs_every_threshold_subset_and_leading_zero_bytes():
    secret = b"\0\0"+bytes(range(30))
    shares = share_secret(secret, threshold=3, count=5)
    for subset in combinations(shares, 3):
        assert reconstruct_secret(subset, 3) == secret
    assert reconstruct_secret(shares, 3) == secret
    with pytest.raises(ValueError, match="Insufficient"):
        reconstruct_secret(shares[:2], 3)


def test_shamir_rejects_duplicates_domain_errors_and_inconsistent_extra_share():
    shares = share_secret(b"x"*32, 3, 5)
    with pytest.raises(ValueError, match="Duplicate"):
        reconstruct_secret([shares[0], shares[0], shares[1]], 3)
    with pytest.raises(ValueError, match="field"):
        reconstruct_secret([(0, 1), shares[0], shares[1]], 3)
    with pytest.raises(ValueError, match="Inconsistent"):
        reconstruct_secret(shares[:3]+[(shares[3][0], (shares[3][1]+1) % PRIME)], 3)
