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
    import hashlib

    pp = LHH.Gen(60)
    a = np.zeros(60, dtype=np.int64)
    b = np.zeros(60, dtype=np.int64)
    a[0] = 123
    b[50] = -77

    # paper notation: LHH.Hash(a + b) == LHH.Mul(LHH.Hash(a), LHH.Hash(b))
    assert LHH.Hash(pp, a + b) == LHH.Mul(LHH.Hash(pp, a), LHH.Hash(pp, b))

    old_collision = np.zeros(60, dtype=np.int64)
    old_collision[0] = 1
    old_collision[50] = -1
    zero = np.zeros(60, dtype=np.int64)

    assert LHH.Hash(pp, old_collision) != LHH.Hash(pp, zero)

    # F1 回归: 生成元的离散对数不可由公开种子派生 (hash-to-curve)。
    # 旧实现把 seed 哈希成标量 a_i 再乘 G (泄露 a_i); 现在应不再是 a_i*G。
    from utils import LHHPublicParams
    seed = LHHPublicParams.seed
    order = LHH.CURVE.order

    def old_scalar(i):
        material = (
            LHH.GENERATOR_DOMAIN
            + len(seed).to_bytes(4, "big")
            + seed
            + int(i).to_bytes(16, "big", signed=False)
        )
        return int.from_bytes(hashlib.sha256(material).digest(), "big") % (order - 1) + 1

    for i in (0, 1, 2):
        new_g = LHH._generator_for(seed, i)
        old_g = LHH.CURVE.generator * old_scalar(i)
        assert new_g != old_g, f"generator {i} leaks discrete log"

    # 旧公开标量碰撞向量 (Δ_0=a_1, Δ_1=-a_0) 在新方案下不应是碰撞。
    delta = np.zeros(60, dtype=object)
    delta[0] = old_scalar(1)
    delta[1] = -old_scalar(0)
    assert LHH.Hash(pp, delta) != LHH.CURVE.generator * 0


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
    cloud_model, plaintext_sum, hash_list = cloud.trigger_final_unmasking_and_averaging()

    expected_sum = np.sum(np.stack(plaintext_models, axis=0), axis=0)
    assert np.array_equal(plaintext_sum, expected_sum)
    # 对齐安全模型: Cloud 不提供规范模型; 共识模型由已验证明文和重构
    assert cloud_model is None
    reconstructed = plaintext_sum.astype(np.float64) / QUANTIZE_SCALE / len(users)
    assert np.allclose(reconstructed, expected_sum / len(users) / QUANTIZE_SCALE)
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
    cloud_model, plaintext_sum, hash_list = cloud.trigger_final_unmasking_and_averaging()

    assert ta.h_mask_all is not None
    assert hash_list[0]["lhh_hash"] is None
    assert np.array_equal(plaintext_sum, model_int)
    assert cloud_model is None
    reconstructed = plaintext_sum.astype(np.float64) / QUANTIZE_SCALE  # n_online = 1
    assert np.allclose(reconstructed, model_int / QUANTIZE_SCALE)
    assert edge.verify_global_result(plaintext_sum, hash_list)


if __name__ == "__main__":
    test_ka_prg_and_signature()
    test_lhh_homomorphism_and_no_generator_reuse_collision()
    test_end_to_end_hvsa_and_tamper_detection()
    test_fast_mode_skips_verification_only()
    print("all crypto primitive tests passed")
