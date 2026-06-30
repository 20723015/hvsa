import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle

import numpy as np

from cloud import CloudServer
from edge import EdgeServer
from ta import TrustedAuthority
from user import User
from utils import KA, LHH, PRG, QUANTIZE_SCALE, SIG


def test_ka_prg_and_signature():
    user_pub, user_priv = KA.gen()
    ta_pub, ta_priv = KA.gen()
    user_seed = KA.agree(user_priv, ta_pub)
    ta_seed = KA.agree(ta_priv, user_pub)

    assert user_seed == ta_seed
    assert len(user_seed) == 32

    mask_a = PRG.mask_int(user_seed, "U-1", 7, 32)
    mask_b = PRG.mask_int(ta_seed, "U-1", 7, 32)
    mask_next_round = PRG.mask_int(ta_seed, "U-1", 8, 32)
    mask_other_user = PRG.mask_int(ta_seed, "U-2", 7, 32)

    assert np.array_equal(mask_a, mask_b)
    assert not np.array_equal(mask_a, mask_next_round)
    assert not np.array_equal(mask_a, mask_other_user)
    assert mask_a.dtype == np.int64
    assert int(mask_a.min()) >= PRG.MASK_LOW
    assert int(mask_a.max()) < PRG.MASK_HIGH

    pub_key, priv_key = SIG.gen()
    msg = b"hvsa-ed25519-test"
    signature = SIG.sign(msg, priv_key)

    assert len(signature) == 64
    assert SIG.verify(msg, signature, pub_key)
    assert not SIG.verify(msg + b"!", signature, pub_key)
    assert not SIG.verify(msg, signature[:-1] + bytes([signature[-1] ^ 1]), pub_key)


def test_lhh_homomorphism_and_no_generator_reuse_collision():
    pp = LHH.Gen(60)
    a = np.zeros(60, dtype=np.int64)
    b = np.zeros(60, dtype=np.int64)
    a[0] = 123
    b[50] = -77

    assert LHH.Hash(pp, a + b) == LHH.Add(LHH.Hash(pp, a), LHH.Hash(pp, b))

    old_collision = np.zeros(60, dtype=np.int64)
    old_collision[0] = 1
    old_collision[50] = -1
    zero = np.zeros(60, dtype=np.int64)

    assert LHH.Hash(pp, old_collision) != LHH.Hash(pp, zero)


def test_end_to_end_hvsa_and_tamper_detection():
    """端到端测试"""
    dim = 8
    round_t = 1
    num_edges = 2
    users_per_edge = 2

    ta = TrustedAuthority(gradient_dim=dim, ta_id="TA-01")
    cloud = CloudServer()
    edges = [EdgeServer(f"ES-{i}", cloud, ta, enable_verification=True) for i in range(num_edges)]

    users = []
    user_edge = {}
    ta_pub = ta.get_public_key_bytes()
    for i in range(num_edges * users_per_edge):
        user = User(f"U-{i}", None)
        ta.register_user(user.id, user.get_public_key_bytes())
        user.register_with_ta(ta_pub)
        users.append(user)
        user_edge[user.id] = edges[i // users_per_edge]

    cloud.start_new_round(round_t, dim, num_edges)
    ta.start_new_round(round_t)
    for edge in edges:
        edge.start_new_round(round_t)

    plaintext_models = []
    for i, user in enumerate(users):
        model_int = np.arange(dim, dtype=np.int64) + i * 100
        plaintext_models.append(model_int)
        masked = model_int + user._generate_ta_mask_int(round_t, dim)
        msg = {"user_id": user.id, "round": round_t, "masked_gradient": masked}
        user_edge[user.id].receive_masked_gradient(pickle.dumps(msg))

    for edge in edges:
        edge.trigger_aggregation_and_forward()
    ta.trigger_global_mask_computation(cloud)
    avg_model, plaintext_sum, hash_list = cloud.trigger_final_unmasking_and_averaging()

    expected_sum = np.sum(np.stack(plaintext_models, axis=0), axis=0)
    assert np.array_equal(plaintext_sum, expected_sum)
    assert np.allclose(avg_model, expected_sum / len(users) / QUANTIZE_SCALE)
    assert all(edge.verify_global_result(plaintext_sum, hash_list) for edge in edges)

    tampered_sum = plaintext_sum.copy()
    tampered_sum[0] += 1
    assert not edges[0].verify_global_result(tampered_sum, hash_list)


def test_fast_mode_skips_verification_only():
    """快速模式测试"""
    dim = 8
    round_t = 3

    ta = TrustedAuthority(gradient_dim=dim, ta_id="TA-01")
    cloud = CloudServer()
    edge = EdgeServer("ES-fast", cloud, ta, enable_verification=False)
    user = User("U-fast", None)
    ta.register_user(user.id, user.get_public_key_bytes())
    user.register_with_ta(ta.get_public_key_bytes())

    cloud.start_new_round(round_t, dim, 1)
    ta.start_new_round(round_t)
    edge.start_new_round(round_t)

    model_int = np.arange(dim, dtype=np.int64)
    masked = model_int + user._generate_ta_mask_int(round_t, dim)
    msg = {"user_id": user.id, "round": round_t, "masked_gradient": masked}
    edge.receive_masked_gradient(pickle.dumps(msg))

    edge.trigger_aggregation_and_forward()
    ta.trigger_global_mask_computation(cloud)
    avg_model, plaintext_sum, hash_list = cloud.trigger_final_unmasking_and_averaging()

    assert ta.h_mask_all is not None
    assert hash_list[0]["lhh_hash"] is None
    assert np.array_equal(plaintext_sum, model_int)
    assert np.allclose(avg_model, model_int / QUANTIZE_SCALE)
    assert edge.verify_global_result(plaintext_sum, hash_list)


if __name__ == "__main__":
    test_ka_prg_and_signature()
    test_lhh_homomorphism_and_no_generator_reuse_collision()
    test_end_to_end_hvsa_and_tamper_detection()
    test_fast_mode_skips_verification_only()
    print("all crypto primitive tests passed")
