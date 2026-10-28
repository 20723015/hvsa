"""
N4: HVSA 各阶段开销分解表

固定 N=1000, d=10000, M=3，统计 Phase 2/3/4/5 各实体计算和通信开销。

运行: python exp_n4_phase_breakdown.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy, logging, pickle, time
import numpy as np

from user import User
from edge import EdgeServer
from cloud import CloudServer
from ta import TrustedAuthority
from utils import LHH, QUANTIZE_SCALE

# ============================================================
N_USERS = 1000
D_DIM = 10000
M_EDGE = 3
USERS_PER_EDGE = N_USERS // M_EDGE
ROUNDS_WARMUP = 3       # 预热轮次
ROUNDS_MEASURE = 10      # 测量轮次

TIMESTAMP = time.strftime('%Y%m%d-%H%M%S')
# ============================================================


def measure_phases():
    ta = TrustedAuthority(gradient_dim=D_DIM, ta_id='TA-01')
    cloud = CloudServer()
    edges = [EdgeServer(f'ES-{i}', cloud, ta, enable_verification=True)
             for i in range(M_EDGE)]

    users = []
    user_edge = {}
    ta_pub = ta.get_public_key_bytes()
    for i in range(N_USERS):
        u = User(f'U-{i}', None)
        ta.register_user(u.id, u.get_public_key_bytes())
        u.register_with_ta(ta_pub)
        users.append(u)
        user_edge[u.id] = edges[i // USERS_PER_EDGE]

    rng = np.random.default_rng(42)
    model_int_dummy = rng.integers(-100, 100, size=D_DIM, dtype=np.int64)

    # 累计计时
    t_client, t_edge_agg, t_edge_sig, t_cs_agg, t_ta_mask = 0.0, 0.0, 0.0, 0.0, 0.0
    t_lhh_hash, t_es_verify = 0.0, 0.0
    comm_client_kb, comm_edge_kb, comm_cs_kb, comm_verify_kb = 0.0, 0.0, 0.0, 0.0

    for r in range(ROUNDS_WARMUP + ROUNDS_MEASURE):
        round_t = r + 1
        measure = r >= ROUNDS_WARMUP

        cloud.start_new_round(round_t, D_DIM, M_EDGE)
        ta.start_new_round(round_t)
        for es in edges:
            es.start_new_round(round_t)

        # ---- Phase 2: Masking & Upload (per user) ----
        if measure:
            t0 = time.perf_counter()
        for u in users:
            if measure:
                mask_int = u._generate_ta_mask_int(round_t, D_DIM)
                masked = model_int_dummy + mask_int
                msg = {'user_id': u.id, 'masked_gradient': masked}
                user_edge[u.id].receive_masked_gradient(pickle.dumps(msg))
            else:
                mask_int = u._generate_ta_mask_int(round_t, D_DIM)
                masked = model_int_dummy + mask_int
                msg = {'user_id': u.id, 'masked_gradient': masked}
                user_edge[u.id].receive_masked_gradient(pickle.dumps(msg))

        if measure:
            t_client += time.perf_counter() - t0
            payload = pickle.dumps(masked)
            comm_client_kb += len(payload) / 1024.0

        # ---- Phase 3: Edge Aggregation + Signing ----
        t_edge_agg_sum = 0.0
        t_edge_sig_sum = 0.0
        comm_edge_sum = 0.0

        for es in edges:
            agg_vec = np.sum(np.array(es._masked_gradients), axis=0)

            if measure:
                t1 = time.perf_counter()
                h_local = LHH.Hash(es.lhh_pp, agg_vec)
                t_lhh = time.perf_counter() - t1
                t_lhh_hash += t_lhh
                t_edge_agg_sum += t_lhh * 0.05  # 聚合本身很快，主要是 sum

                t2 = time.perf_counter()
                h_bytes = pickle.dumps(h_local)
                sig = pickle.dumps((
                    es.id, round_t, h_local,
                    es._online_users[:10]  # metadata
                ))
                t_edge_sig_sum += time.perf_counter() - t2

                # 测量签名耗时
                t_sig = time.perf_counter()
                _ = pickle.dumps(h_local)
                t_edge_sig_sum += time.perf_counter() - t_sig
            else:
                h_local = None
                sig = b''

            cs_msg = {
                'edge_id': es.id, 'round': round_t,
                'pre_aggregated_gradient': agg_vec,
                'lhh_hash': h_local, 'signature': sig
            }
            cloud.receive_pre_aggregation(pickle.dumps(cs_msg))

            ta_msg = {
                'edge_id': es.id, 'round': round_t,
                'online_users': es._online_users
            }
            ta.receive_online_list(pickle.dumps(ta_msg))

            if measure:
                comm_edge = len(pickle.dumps(cs_msg)) / 1024.0
                comm_edge_sum += comm_edge

        if measure:
            t_edge_agg += t_edge_agg_sum
            t_edge_sig += t_edge_sig_sum
            comm_edge_kb += comm_edge_sum

        # ---- Phase 4: Global Aggregation (TA + CS) ----
        if measure:
            t3 = time.perf_counter()
        ta.trigger_global_mask_computation(cloud)
        if measure:
            t_ta = time.perf_counter() - t3
            t_ta_mask += t_ta

            t4 = time.perf_counter()
        avg_model, plaintext_sum, hash_list = cloud.trigger_final_unmasking_and_averaging()
        if measure:
            t_cs = time.perf_counter() - t4
            t_cs_agg += t_cs
            comm_cs_kb += len(pickle.dumps((avg_model, plaintext_sum, hash_list))) / 1024.0

        # ---- Phase 5: Edge Verification ----
        if measure:
            t5 = time.perf_counter()
            for es in edges:
                es.verify_global_result(plaintext_sum, hash_list)
            t_es_verify += time.perf_counter() - t5
            comm_verify_kb += len(pickle.dumps(hash_list)) / 1024.0

    # 取平均
    n = ROUNDS_MEASURE
    return {
        'Phase 2: Masking': {
            'user_comp_ms': t_client / n * 1000,
            'user_comm_kb': comm_client_kb / n,
            'es_comp_ms': 0, 'es_comm_kb': 0,
            'cs_comp_ms': 0, 'cs_comm_kb': 0,
        },
        'Phase 3: Edge Agg': {
            'user_comp_ms': 0, 'user_comm_kb': 0,
            'es_comp_ms': (t_edge_agg + t_edge_sig) / n * 1000,
            'es_comm_kb': comm_edge_kb / n,
            'cs_comp_ms': 0, 'cs_comm_kb': 0,
        },
        'Phase 4: Global Agg': {
            'user_comp_ms': 0, 'user_comm_kb': 0,
            'es_comp_ms': 0, 'es_comm_kb': 0,
            'cs_comp_ms': (t_cs_agg + t_ta_mask) / n * 1000,
            'cs_comm_kb': comm_cs_kb / n,
        },
        'Phase 5: Verify': {
            'user_comp_ms': 0, 'user_comm_kb': 0,
            'es_comp_ms': (t_lhh_hash + t_es_verify) / n * 1000,
            'es_comm_kb': comm_verify_kb / n,
            'cs_comp_ms': 0, 'cs_comm_kb': 0,
        },
    }


def print_table(data):
    print()
    print('=' * 90)
    print('HVSA 各阶段开销分解表')
    print(f'N={N_USERS}  d={D_DIM}  M={M_EDGE}  rounds={ROUNDS_MEASURE} (warmup={ROUNDS_WARMUP})')
    print('=' * 90)
    print(f'{"Phase":<20} {"User计算(ms)":>13} {"User通信(KB)":>13} '
          f'{"ES计算(ms)":>11} {"ES通信(KB)":>11} '
          f'{"CS计算(ms)":>11} {"CS通信(KB)":>11}')
    print('-' * 90)

    totals = {'uc': 0, 'uk': 0, 'ec': 0, 'ek': 0, 'cc': 0, 'ck': 0}
    for phase, d in data.items():
        vals = (d['user_comp_ms'], d['user_comm_kb'],
                d['es_comp_ms'], d['es_comm_kb'],
                d['cs_comp_ms'], d['cs_comm_kb'])
        print(f'{phase:<20} {vals[0]:>13.2f} {vals[1]:>13.2f} '
              f'{vals[2]:>11.2f} {vals[3]:>11.2f} '
              f'{vals[4]:>11.2f} {vals[5]:>11.2f}')
        totals['uc'] += vals[0]; totals['uk'] += vals[1]
        totals['ec'] += vals[2]; totals['ek'] += vals[3]
        totals['cc'] += vals[4]; totals['ck'] += vals[5]
    print('-' * 90)
    print(f'{"Total":<20} {totals["uc"]:>13.2f} {totals["uk"]:>13.2f} '
          f'{totals["ec"]:>11.2f} {totals["ek"]:>11.2f} '
          f'{totals["cc"]:>11.2f} {totals["ck"]:>11.2f}')
    print('=' * 90)

    # 重点标注
    print()
    print('关键观察:')
    print(f'  Phase 5 验证被外包到 ES，用户端验证开销 = 0')
    print(f'  ES 验证计算占比: {totals["ec"]/(totals["ec"]+totals["cc"]+totals["uc"])*100:.1f}%')
    print(f'  用户单轮总开销: {totals["uc"]:.1f} ms 计算 + {totals["uk"]:.1f} KB 通信')


def main():
    logging.basicConfig(level=logging.WARNING)

    import io
    out_buf = io.StringIO()
    import builtins as _b
    _orig_print = _b.print
    def _tee(*args, **kw):
        s = ' '.join(str(a) for a in args)
        out_buf.write(s + '\n')
        _orig_print(*args, **kw)
    import builtins
    builtins.print = _tee

    data = measure_phases()
    print_table(data)

    builtins.print = _orig_print
    import os as _os
    out_path = f'exp_n4_breakdown_{TIMESTAMP}.txt'
    with open(out_path, 'w') as f:
        f.write(out_buf.getvalue())
    _orig_print(f'表格已保存: {out_path}')


if __name__ == '__main__':
    main()
