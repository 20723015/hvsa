"""临时验证 P0-3: 攻击检测率在小维度下应为 100%."""
import hashlib
import numpy as np, pickle
from ta import TrustedAuthority
from edge import EdgeServer
from cloud import CloudServer
from point2_types import GroupPackage
from utils import PRG, LHH

DIM = 8
SEED = b's'

def make_cluster(n_edges=2, users_per_edge=3):
    ta = TrustedAuthority(DIM)
    ta.point2_reset()
    cloud = CloudServer()
    cloud._gradient_dim = DIM
    cloud.point2_reset(np.zeros(DIM))
    edges = []
    for i in range(n_edges):
        es = EdgeServer(f'e{i}', cloud, ta, enable_verification=True,
                        gamma_vcm=1.0)
        edges.append(es)
    return ta, cloud, edges

def make_package(es, users, base_v, token, base_hash, sample=100, seed=SEED):
    es.p2_start_package(base_version=base_v, base_hash=base_hash, package_token=token)
    plain = {}
    for u in users:
        ta._user_secrets[u] = seed
        w = np.random.randint(-100, 100, DIM).astype(np.float64)
        plain[u] = w
        m = PRG.mask_int(seed, u, token, DIM)
        masked = (w * sample).astype(np.int64) + m
        es.p2_receive_masked_gradient(
            pickle.dumps({'user_id': u, 'masked_gradient': masked}),
            token, base_v, sample_size=sample)
    return es.p2_seal_group_package(), plain

ta, cloud, edges = make_cluster()
base_hash = hashlib.sha256(
    cloud.p2_global_model_history[0].tobytes()).digest()
pkgs = []
for i, es in enumerate(edges):
    users = [f'e{i}-u{j}' for j in range(3)]
    pkg, _ = make_package(es, users, base_v=0, token=f'e{i}:v0:p0',
                          base_hash=base_hash)
    assert pkg is not None
    pkgs.append(pkg)
    cloud.p2_receive_group_package(pkg, arrival_tick=1)

selected = cloud.p2_select_checkpoint(K=2, s_max=10, current_tick=1)
assert len(selected) == 2, f"selected={len(selected)}"
M_Q, H_mask = ta.p2_build_checkpoint_mask(selected)
raw = cloud.p2_recover_raw_sum(selected, M_Q)
corrected, final, comp = cloud.p2_apply_vcm_and_average(selected, raw, gamma_vcm=1.0)

def make_bundle(**kw):
    b = {
        'checkpoint_version': 0,
        'policy_config_digest': es0._policy_digest,
        'selected_packages': selected,
        'raw_sum_int': raw,
        'M_Q': M_Q,
        'H_mask_raw': H_mask,
        'final_model': final,
        'gamma_vcm': 1.0,
        'global_model_history': cloud.p2_global_model_history,
    }
    b.update(kw)
    return b

es0 = edges[0]
honest = es0.p2_verify_checkpoint(make_bundle())
print('honest bundle passes:', honest)
assert honest, "诚实路径必须通过"

# tamper_global: 篡改 W_sum^raw
tampered_raw = raw.copy(); tampered_raw[0] += 1
_, tf, _ = cloud.p2_apply_vcm_and_average(selected, tampered_raw, gamma_vcm=1.0)
det = not es0.p2_verify_checkpoint(make_bundle(raw_sum_int=tampered_raw, final_model=tf))
print('tamper_global detected:', det); assert det

# tamper_manifest: 注入伪造组包
fake = GroupPackage(
    edge_id='fake_edge', package_token='fake:pkg', base_version=0,
    base_hash=b'\x00', online_users=['fake_u'], n_k=1, S_k=100,
    masked_sum_int=np.zeros(DIM, dtype=np.int64),
    H_k=LHH.Hash(ta.lhh_pp, np.zeros(DIM, dtype=np.int64)),
    signature=b'\x00', created_tick=1)
det = not es0.p2_verify_checkpoint(make_bundle(selected_packages=selected + [fake]))
print('tamper_manifest detected:', det); assert det

# drop_replace_pkg: 丢弃一个组包 (raw 诚实, selected 缺包)
det = not es0.p2_verify_checkpoint(make_bundle(selected_packages=selected[:-1]))
print('drop_replace_pkg detected:', det); assert det

print('ALL P0-3 CHECKS PASSED')
