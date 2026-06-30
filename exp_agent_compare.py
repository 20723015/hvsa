"""
实验2: 硬编码 vs Agent编排 对比
不调API，全部Mock。快速出结果。
"""
import os,sys,io,time,copy,logging,json
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np, matplotlib.pyplot as plt
import torch; from torch.utils.data import DataLoader
import models; from datasets import load_data,distribute_data
from user import User; from ta import TrustedAuthority
from agent_async import AgentAuditor,AgentClient,AgentCoordinator,AgentEdge
from orchestrate_agent import OrchestrationRuleEngine,patch_auditor_for_orchestration

for h in logging.root.handlers[:]:logging.root.removeHandler(h)
logging.basicConfig(level=logging.WARNING)

DATASET='MNIST'; CFG={'model':'CNN','rounds':30,'local_epochs':1,'lr':0.01,'samples_per_user':150,'batch_size':32}
NUM_EDGE=2; USERS_PER_EDGE=4; TOTAL=NUM_EDGE*USERS_PER_EDGE
MAX_TICK=30; K_USER=2; Q_ES=2

def evaluate(flat,meta,loader):
    m=models.get_model(DATASET,CFG['model'])
    models.unflatten_parameters(m,flat,meta)
    return models.evaluate(m,loader)

def run_experiment(use_agent:bool,drop_rate:float,label:str):
    train_ds,test_ds=load_data(DATASET); test_ldr=DataLoader(test_ds,256,False)
    loaders=distribute_data(train_ds,TOTAL,CFG['batch_size'],iid=True,max_samples_per_user=CFG['samples_per_user'])
    init_m=models.get_model(DATASET,CFG['model']); v,meta=models.flatten_parameters(init_m)

    ta=TrustedAuthority(v.shape[0],'TA-01')
    auditor=AgentAuditor(); patch_auditor_for_orchestration(auditor)
    orch=OrchestrationRuleEngine(quorum_q_es=Q_ES) if use_agent else None

    coord=AgentCoordinator(ta,v,async_weight=0.6,quorum_q_es=Q_ES,orchestrator=orch)
    edges=[AgentEdge(f'ES-{i}',ta,coord,auditor,K_USER,5,False)for i in range(NUM_EDGE)]
    edgemap={e.id:e for e in edges}

    clients=[]; rng=np.random.default_rng(7)
    for idx in range(TOTAL):
        u=User(f'U-{idx}',copy.deepcopy(init_m))
        ta.register_user(u.id,u.get_public_key_bytes())
        u.register_with_ta(ta.get_public_key_bytes())
        clients.append(AgentClient(u,loaders[idx],rng,(1,3),drop_rate,llm_reasoner=None))

    acc_hist=[]; vec=v.copy()
    for tick in range(MAX_TICK):
        gv,gver=coord.get_global_snapshot()
        for cl in clients:
            if cl.is_idle():
                e=edges[clients.index(cl)//USERS_PER_EDGE]
                wt,bv=e.assign_window(gver,tick)
                cl.start_task(gv,meta,e.id,wt,bv,CFG['local_epochs'],CFG['lr'])
        for cl in clients:
            st,up,di=cl.tick()
            if st=='upload'and up:edgemap[up.edge_id].receive_upload(up,tick)
            elif st=='drop'and di:eid,wt=di;edgemap[eid].note_dropout(wt)
        for e in edges:e.tick(tick)
        commits=coord.process_quorum(tick,edges)
        for cm in commits:
            if cm.verification_passed and cm.batch_size>0:
                if coord.commit_count%2==0:
                    acc,loss=evaluate(coord.global_model_vector,meta,test_ldr)
                    acc_hist.append((coord.commit_count,acc))
    return acc_hist

print('='*60)
print('实验2: 硬编码 vs Agent编排 (全部Mock)')
print(f'配置: {TOTAL}用户 {NUM_EDGE}Edge K_user={K_USER} Q_ES={Q_ES}')
print('='*60)

results={}
for use_agent in [False,True]:
    for drop in [0.0,0.1,0.2]:
        label=f"{'Agent' if use_agent else 'Hard'}_{int(drop*100)}"
        print(f'\n[{label}] 跑实验...')
        t0=time.perf_counter()
        h=run_experiment(use_agent,drop,label)
        elapsed=time.perf_counter()-t0
        results[label]=h
        if h:
            accs=[a for _,a in h]
            print(f'  提交: {len(h)}次  最终准确率: {accs[-1]:.1f}%  耗时: {elapsed:.0f}s')
        else:
            print(f'  无有效提交')

# 绘图
fig,axes=plt.subplots(1,3,figsize=(16,5))
for di,drop in enumerate([0.0,0.1,0.2]):
    ax=axes[di]
    for mode,sty,clr in [('Hard','--','blue'),('Agent','-','red')]:
        k=f'{mode}_{int(drop*100)}'
        if k in results and results[k]:
            x=[a[0]for a in results[k]];y=[a[1]for a in results[k]]
            ax.plot(x,y,sty,color=clr,lw=2,label=f'{mode}')
    ax.set_title(f'Dropout {int(drop*100)}%');ax.set_xlabel('Commits');ax.set_ylabel('Accuracy(%)')
    ax.legend();ax.grid(linestyle='--',alpha=0.5)

plt.tight_layout();plt.savefig('agent_vs_hardcoded.png',dpi=200)
print('\n图表已保存: agent_vs_hardcoded.png')

# 总结
print('\n总结:')
for drop in [0.0,0.1,0.2]:
    ha=[a for _,a in results.get(f'Hard_{int(drop*100)}',[])]
    ag=[a for _,a in results.get(f'Agent_{int(drop*100)}',[])]
    if ha and ag:
        print(f'  掉线率{int(drop*100)}%: Hard={ha[-1]:.1f}%  Agent={ag[-1]:.1f}%')
