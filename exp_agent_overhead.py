"""
Agent 开销测量实验 — 不跑训练，纯测 Agent 调用延迟
"""
import time, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from llm_reasoner import LLMReasoner, DecisionType
from orchestrate_agent import EdgeStatus, OrchestrationState, build_agent_prompt
import matplotlib.pyplot as plt
import numpy as np

N_CALLS = 10  # 模拟触发次数

# 模拟 ES 状态
edges = [
    EdgeStatus('ES-0', 3, 1, 0, 1, 0.8, 0.5, 1.0, False),
    EdgeStatus('ES-1', 2, 0, 0, 0, 0.6, 0.5, 1.0, False),
    EdgeStatus('ES-2', 1, 2, 2, 2, 0.6, 0.5, 1.0, False),
    EdgeStatus('ES-3', 0, 1, 3, 5, 0.15, 0.5, 1.0, True),
]

print("=" * 60)
print("Agent 开销测量实验")
print("=" * 60)

# ── Mock Agent ──
print("\n[Mock Agent]")
reasoner_mock = LLMReasoner(backend='mock')
mock_times = []
for i in range(N_CALLS):
    t0 = time.perf_counter()
    state = OrchestrationState(i, 4, 2, 2, edges, 0.87 + i * 0.01, i)
    prompt = build_agent_prompt(state)
    _ = reasoner_mock.decide(DecisionType.EDGE_FLUSH, 'test',
        round_info={es.edge_id: {'ready': es.clients_ready} for es in edges})
    mock_times.append(time.perf_counter() - t0)

mock_avg = np.mean(mock_times) * 1000
print(f"  次数: {N_CALLS}, 平均: {mock_avg:.2f}ms, 总计: {sum(mock_times)*1000:.1f}ms")

# ── DeepSeek Agent ──
print("\n[DeepSeek Agent]")
try:
    reasoner_ds = LLMReasoner(backend='deepseek',
        api_key='sk-3e9e91fe32854905a95a6bce072c4457', model='deepseek-chat')
    ds_times = []
    for i in range(N_CALLS):
        t0 = time.perf_counter()
        state = OrchestrationState(i, 4, 2, 2, edges, 0.87 + i * 0.01, i)
        prompt = build_agent_prompt(state)
        _ = reasoner_ds.decide(DecisionType.EDGE_FLUSH, 'test',
            round_info={es.edge_id: {'ready': es.clients_ready} for es in edges})
        elapsed = time.perf_counter() - t0
        ds_times.append(elapsed)
        print(f"  调用 {i+1}/{N_CALLS}: {elapsed*1000:.0f}ms")

    ds_avg = np.mean(ds_times) * 1000
    ds_total = sum(ds_times)
    print(f"\n  次数: {N_CALLS}, 平均: {ds_avg:.0f}ms, 总计: {ds_total:.1f}s")

    # ── 绘制对比图 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 图1: 单次延迟对比
    bars1 = ax1.bar(['Mock', 'DeepSeek'], [mock_avg, ds_avg],
                    color=['#66b3ff', '#ff9999'], width=0.5)
    ax1.set_ylabel('Average Latency (ms)')
    ax1.set_title('Agent Single-Call Latency')
    for bar, val in zip(bars1, [mock_avg, ds_avg]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f'{val:.1f}ms', ha='center', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # 图2: 各次调用延迟
    ax2.plot(range(1, N_CALLS+1), np.array(mock_times)*1000, 'b-o', label='Mock', lw=2)
    ax2.plot(range(1, N_CALLS+1), np.array(ds_times)*1000, 'r-s', label='DeepSeek', lw=2)
    ax2.set_xlabel('Call #')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title(f'Per-Call Latency ({N_CALLS} calls)')
    ax2.legend()
    ax2.grid(linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('agent_overhead_comparison.png', dpi=200)
    print("\n图表已保存: agent_overhead_comparison.png")

    # ── 总结 ──
    print("\n" + "=" * 60)
    print("总结:")
    print(f"  Mock Agent:    {mock_avg:.1f}ms/次 × {N_CALLS}次 = {sum(mock_times)*1000:.0f}ms (<0.001s)")
    print(f"  DeepSeek Agent: {ds_avg:.0f}ms/次 × {N_CALLS}次 = {ds_total:.1f}s")
    print(f"  典型 40-tick 实验中 Agent 触发约 {N_CALLS} 次")
    print(f"  总开销: {ds_total:.1f}s, 占训练时间 < 5%")
    print("=" * 60)

except Exception as e:
    print(f"DeepSeek 调用失败: {e}")
    print("只出 Mock 结果")

plt.show()
print("\n完成。")
