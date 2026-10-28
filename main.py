import logging
import pickle
import hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader
import copy
import sys

try:
    from user import User
    from edge import EdgeServer
    from cloud import CloudServer
    from ta import TrustedAuthority
except ImportError as e:
    logging.error(f"导入失败: {e}")
    sys.exit(1)

try:
    import models
    import datasets
    from datasets import load_data, distribute_data
    from utils import QUANTIZE_SCALE
except ImportError as e:
    logging.error(f"导入失败: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

DATASET = 'MNIST'
MODEL_NAME = 'CNN'
LOCAL_EPOCHS = 5
LEARNING_RATE = 0.015
BATCH_SIZE = 32

NUM_EDGE_SERVERS = 2
NUM_USERS_PER_CLUSTER = 2
TOTAL_FL_ROUNDS = 1
USER_DROPOUT_RATE = 0.1
ENABLE_VERIFICATION_MODE = False


def run_simulation():
    logging.info(f"--- 1. 架构初始化开始: {NUM_EDGE_SERVERS} 个 ES, 每个 {NUM_USERS_PER_CLUSTER} 个用户 ---")

    TOTAL_USERS = NUM_EDGE_SERVERS * NUM_USERS_PER_CLUSTER

    train_dataset, test_dataset = load_data(DATASET)
    all_dataloaders = distribute_data(train_dataset, TOTAL_USERS, BATCH_SIZE, iid=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    global_model = models.get_model(DATASET, MODEL_NAME)
    tmp_model = models.get_model(DATASET, MODEL_NAME)

    initial_weights, model_meta = models.flatten_parameters(tmp_model)
    GRADIENT_DIM = initial_weights.shape[0]
    logging.info(f"模型参数已展平。总参数量: {GRADIENT_DIM}")

    # 初始化实体 (单TA)
    ta = TrustedAuthority(gradient_dim=GRADIENT_DIM, ta_id="TA-01")
    cs = CloudServer()

    edge_servers = []
    for i in range(NUM_EDGE_SERVERS):
        es = EdgeServer(id=f"edge_{i}", cloud_server=cs,
                        ta=ta, enable_verification=ENABLE_VERIFICATION_MODE)
        edge_servers.append(es)

    all_users = []
    user_to_edge_server = {}

    for i in range(NUM_EDGE_SERVERS):
        current_edge_server = edge_servers[i]
        for j in range(NUM_USERS_PER_CLUSTER):
            user_id = str(1001 + i * NUM_USERS_PER_CLUSTER + j)
            user_model = copy.deepcopy(tmp_model)
            user = User(id=user_id, model=user_model)
            all_users.append(user)
            user_to_edge_server[user_id] = current_edge_server

    logging.info("--- (SA) 正在向 TA 注册所有用户... ---")
    ta_pub = ta.get_public_key_bytes()
    for user in all_users:
        user.register_with_ta(ta_pub)
        ta.register_user(user.id, user.get_public_key_bytes())
    logging.info("--- (SA) 所有用户向 TA 注册完毕。 ---")

    logging.info(f"--- 1. 架构初始化完成。 ---")

    logging.info("--- 2. 标准联邦训练开始 ---")

    global_weights_vector, _ = models.flatten_parameters(global_model)

    for r in range(TOTAL_FL_ROUNDS):
        logging.info(f"\n--- FL 全局回合 {r + 1}/{TOTAL_FL_ROUNDS} ---")

        cs.start_new_round(r, GRADIENT_DIM, expected_edge_servers=NUM_EDGE_SERVERS)
        ta.start_new_round(r)
        # 本轮训练基模型 W(t) 的哈希, 供 ES 签名绑定 (对齐 Sec.3.3.3 meta_k)
        base_hash = hashlib.sha256(
            np.asarray(global_weights_vector, dtype=np.float64).tobytes()
        ).digest()
        for es in edge_servers:
            es.start_new_round(r, base_model_hash=base_hash)

        logging.info(f"--- (SA) 回合 {r + 1} (阶段 2) 客户端训练与上传 ---")
        active_online_users = []

        for i, user in enumerate(all_users):
            loader = all_dataloaders[i]

            if np.random.rand() < USER_DROPOUT_RATE:
                logging.info(f"  User {user.id} (R{r}) 随机掉线。")
                continue

            active_online_users.append(user)

            user.set_global_model(global_weights_vector, model_meta)
            user.local_train(loader, LOCAL_EPOCHS, LEARNING_RATE)

            user_es = user_to_edge_server[user.id]
            user.mask_and_upload_model(r, user_es)

        logging.info(f"--- (SA) 回合 {r + 1} (阶段 3) 边缘服务器聚合与转发 ---")
        for es in edge_servers:
            es.trigger_aggregation_and_forward()

        logging.info(f"--- (SA) 回合 {r + 1} (阶段 4a) TA 计算全局掩码 ---")
        ta.trigger_global_mask_computation(cs)

        logging.info(f"--- (SA) 回合 {r + 1} (阶段 4b) 云服务器解掩码 ---")
        result_tuple = cs.trigger_final_unmasking_and_averaging()
        _, g_sum_plaintext, h_all_aggregated = result_tuple

        if g_sum_plaintext is None:
            logging.warning(f"回合 {r + 1} 失败 (数据不足或其他错误)，跳过更新。")
            continue

        logging.info(f"--- (SA) 回合 {r + 1} (阶段 5) 边缘服务器验证 ---")
        validation_passed = True
        for es in edge_servers:
            if not es.verify_global_result(g_sum_plaintext, h_all_aggregated):
                validation_passed = False
                logging.error(f"严重警告: EdgeServer {es.id} 验证失败!")

        if not validation_passed:
            logging.error("本轮验证失败，拒绝更新全局模型。")
            continue

        # 对齐安全模型 (Sec.3.3.5): Cloud 不提供规范模型; ES 验证通过后,
        # 共识模型由已验证明文和 W_sum^raw 按在线数 n_online 本地确定性重构。
        n_online = sum(len(es._online_users) for es in edge_servers)
        global_weights_vector = (
            g_sum_plaintext.astype(np.float64) / QUANTIZE_SCALE / n_online
        )
        models.unflatten_parameters(global_model, global_weights_vector, model_meta)
        logging.info(f"--- (FL) 全局模型已更新 ---")

        accuracy, loss = models.evaluate(global_model, test_loader)
        logging.info(f"--- (FL 全局评估) 回合 {r + 1} 结束。全局模型准确率: {accuracy:.2f}% , 损失: {loss:.4f} ---")


if __name__ == "__main__":
    run_simulation()
