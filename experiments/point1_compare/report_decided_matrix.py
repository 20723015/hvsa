"""Derive publication figures and a bounded report from the complete cost batch."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import datetime
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/"CODEX.md").is_file())
sys.path.insert(0,str(ROOT))
from experiments.point1_compare.artifacts import digest,dump_json
from experiments.point1_compare.run_decided_batch import POINTS,SCHEMES,cost_scope_problems
from experiments.point1_compare.validate_results import validate

ORDER=["hvsa","gpvo_zn2_fresh_prime_e","vlmrs_fullgraph_public_d"]
NAMES={ORDER[0]:"HVSA",ORDER[1]:"GPVO (declared)",ORDER[2]:"VLMRS (declared)"}
CN={ORDER[0]:"HVSA",ORDER[1]:"GPVO本项目实例化",ORDER[2]:"VLMRS本项目实例化"}


def stats(values):
    q=np.percentile(values,[25,50,75])
    return {"count":len(values),"median":float(q[1]),"q25":float(q[0]),"q75":float(q[2]),
            "min":float(min(values)),"max":float(max(values))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    batch_path=Path(args.batch).resolve()
    journal=json.loads(batch_path.read_text(encoding="utf-8"))
    if journal["status"]!="completed_decided_minimum_matrix" or len(journal["jobs"])!=4:
        raise ValueError("All four cost jobs must finish before matrix analysis")
    if journal.get("continued_from"):
        origin=journal["continued_from"]
        if (digest(Path(origin["path"]))!=origin["sha256"]
                or digest(batch_path.parent/"parent_batch_snapshot.json")!=origin["sha256"]
                or digest(batch_path.parent/"continuation_source.py")!=journal["continuation_source_sha256"]):
            raise ValueError("Continuation lineage or source snapshot changed")
    records,sources,initialization=[],{},[]
    for job in journal["jobs"]:
        directory=Path(job["run_dir"])
        if job["status"]!="completed" or digest(directory/"manifest.json")!=job["manifest_sha256"]:
            raise ValueError("Job status or manifest changed: "+job["name"])
        problems=validate(directory)
        if problems: raise ValueError(repr(problems))
        manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        corr=json.loads((directory/"correctness.json").read_text(encoding="utf-8"))
        if cost_scope_problems(manifest,corr,job["point"]): raise ValueError("Wrong cost scope")
        if manifest["decisions_sha256"]!=journal["decisions_sha256"]:
            raise ValueError("Decision version differs")
        for relative,expected in journal["critical_source_sha256"].items():
            archived=directory/manifest["source_archive"]/relative
            if digest(archived)!=expected:
                raise ValueError("Protocol source differs within matrix: "+relative)
        sources[str(directory)]={"manifest_sha256":job["manifest_sha256"],"source_sha256":manifest["source_sha256"],
                                  "output_sha256":manifest["output_sha256"],"gpvo_public_parameters":manifest["gpvo_public_parameters"],
                                  "lib_versions":manifest["lib_versions"]}
        n,m,d=job["point"]
        for item in corr["initialization"]:
            initialization.append({"N":n,"M":None if item["scheme"]==ORDER[2] else m,"d":d,**item,
                                   "total_work_s":sum(item["role_ms"].values())/1000,"source_run":str(directory)})
        for scheme in ORDER:
            all_obs=[o for o in corr["observations"] if o["scheme"]==scheme]
            observed=[o for o in all_obs if not o["warmup"]]
            if len(all_obs)!=13 or len(observed)!=10 or len({o["input_sha256"] for o in all_obs})!=13:
                raise ValueError("Repeated or incomplete round inputs")
            metrics={"client_ms_per_selected":[o["role_ms"]["user"]/n for o in observed],
                     "system_work_s":[sum(o["role_ms"].values())/1000 for o in observed],
                     "wall_s":[o["wall_ms"]/1000 for o in observed],
                     "traffic_mib":[o["total_sent_bytes"]/2**20 for o in observed]}
            metrics.update({role+"_work_s":[o["role_ms"][role]/1000 for o in observed] for role in ["user","edge","ta","cloud"]})
            records.append({"N":n,"M":None if scheme==ORDER[2] else m,"d":d,"scheme":scheme,
                "source_run":str(directory),"warmup_rounds":3,"measured_rounds":10,
                "metrics":{name:stats(values) for name,values in metrics.items()},
                "observations":[{k:o[k] for k in ["round","input_sha256","snapshot_sha256","wall_ms","role_ms","total_sent_bytes"]} for o in observed]})
    if {(r["N"],r["d"],r["scheme"]) for r in records}!={(n,d,s) for n,m,d in POINTS for s in ORDER}:
        raise ValueError("Cost matrix workpoint coverage differs")
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    dump_json(output/"analysis.json",{"artifact_kind":"analysis_of_existing_evidence","records":records,
        "initialization":initialization,"sources":sources,"batch":{"path":str(batch_path),"sha256":digest(batch_path)},
        "executed_rounds":156,"warmup_rounds":36,"measured_rounds":120,
        "definition":"All-client work/N_selected; system work is sum of exclusive role work, not deployed latency."})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
    colors=["#2474A6","#DE8A25","#319578"]
    chart_specs=[("client_cost","client_ms_per_selected","Client computation","All-client work / selected clients (ms)",True),
                 ("system_work","system_work_s","Total protocol computation","Sum of exclusive role work (s)",True),
                 ("communication","traffic_mib","Total protocol communication","Serialized envelopes per round (MiB)",False)]
    for stem,metric,title,ylabel,log in chart_specs:
        fig,ax=plt.subplots(figsize=(9,5.4))
        positions=np.arange(4)
        for si,scheme in enumerate(ORDER):
            values=[next(r for r in records if r["N"]==n and r["d"]==d and r["scheme"]==scheme)["metrics"][metric] for n,m,d in POINTS]
            med=np.array([v["median"] for v in values]);low=med-np.array([v["q25"] for v in values]);high=np.array([v["q75"] for v in values])-med
            ax.bar(positions+(si-1)*.25,med,width=.23,color=colors[si],label=NAMES[scheme],
                   yerr=np.stack([low,high]),capsize=3,error_kw={"linewidth":1})
        if log: ax.set_yscale("log")
        else: ax.set_ylim(bottom=0)
        ax.set_xticks(positions,[f"N={n}\nd={d}" for n,m,d in POINTS])
        ax.set_ylabel(ylabel);fig.suptitle(title,y=.98,fontsize=13)
        ax.legend(loc="upper center",bbox_to_anchor=(.5,1.12),ncol=3,frameon=False,fontsize=9)
        ax.grid(axis="y",alpha=.22);ax.set_axisbelow(True)
        fig.text(.5,.075,"Median and IQR of 10 measured rounds; 3 warmups excluded.",ha="center",fontsize=9)
        fig.text(.5,.035,"Hierarchical M=5; VLMRS has no edge layer. Declared implementations differ in security and backend.",ha="center",fontsize=8)
        fig.subplots_adjust(left=.12,right=.985,bottom=.20,top=.84)
        fig.savefig(output/(stem+".png"),dpi=200)
        fig.savefig(output/(stem+".svg"))
        plt.close(fig)
    lines=["# 三方案明确实例化：最小正式成本矩阵","",
        "日期：2026-10-20。按已授权的实例化配置，完成N=20/50、维度1000/7850四个工作点；分层方案M=5，VLMRS为平面方案，M不适用。每方案每点3次预热+10次正式观测，共156个完整回合（36预热、120正式），全部逐维正确且完整验证通过。", "",
        "三方案在同一工作点共享13份归档输入和实际参与集合，每轮轮换执行顺序。本文所有统计只取10次正式观测，不合并此前257维试跑、旧HVSA单方案N扫描或其他源码版本。N20点保留10月19日完整运行，N50点于10月20日继续执行；中断点从新状态重跑，旧中断预热前缀与暂停等待不计入。协议核心源码在四个运行中与批次冻结摘要相同，完整原始记录已核验。不同工作点采集时段不同，跨N差异不能全部归因于人数。", "",
        "## 主要观察", ""]
    gpvo_ratios={metric:[] for metric in ["client_ms_per_selected","system_work_s","traffic_mib"]}
    communication_tradeoffs=[]
    for n,m,d in POINTS:
        own=next(r for r in records if r["N"]==n and r["d"]==d and r["scheme"]==ORDER[0])["metrics"]
        gpvo=next(r for r in records if r["N"]==n and r["d"]==d and r["scheme"]==ORDER[1])["metrics"]
        vlmrs=next(r for r in records if r["N"]==n and r["d"]==d and r["scheme"]==ORDER[2])["metrics"]
        for metric in gpvo_ratios:
            gpvo_ratios[metric].append(own[metric]["median"]/gpvo[metric]["median"])
        communication_ratio=own["traffic_mib"]["median"]/vlmrs["traffic_mib"]["median"]
        if communication_ratio>1:
            communication_tradeoffs.append(f"N={n}、维度{d}时，HVSA通信量为{own['traffic_mib']['median']:.4f}MiB，VLMRS为{vlmrs['traffic_mib']['median']:.4f}MiB，HVSA高{(communication_ratio-1)*100:.2f}%")
    labels={"client_ms_per_selected":"客户端均摊计算","system_work_s":"全体角色计算","traffic_mib":"全网通信"}
    for metric,values in gpvo_ratios.items():
        lines.append(f"- 在四个实测点，HVSA/{CN[ORDER[1]]}的{labels[metric]}中位数比值为{min(values):.4f}—{max(values):.4f}；仅对应本文冻结配置。")
    for tradeoff in communication_tradeoffs:
        lines.append("- "+tradeoff+"。因此不能写成所有成本指标均领先。")
    lines += ["", "## 客户端、系统工作与通信", "",
        "表中客户端为每轮全部客户端互斥工作除以所选人数N，再取10轮中位数；它不是单个客户端的延迟分位数。系统工作为每轮各角色互斥工作之和；串行回放包含本地驱动开销，不是部署网络时延。IQR为[Q25,Q75]，不代表置信区间。", "",
        "| N | 维度 | 实现 | 客户端/ms每人每轮，中位数[IQR] | 系统工作/s | 串行回放/s，中位数[IQR] | 封装/MiB每轮 |", "|---:|---:|---|---:|---:|---:|---:|"]
    for r in records:
        a=r["metrics"];c=a["client_ms_per_selected"];w=a["wall_s"]
        lines.append(f"| {r['N']} | {r['d']} | {CN[r['scheme']]} | {c['median']:.4f} [{c['q25']:.4f},{c['q75']:.4f}] | {a['system_work_s']['median']:.3f} | {w['median']:.3f} [{w['q25']:.3f},{w['q75']:.3f}] | {a['traffic_mib']['median']:.4f} |")
    for stem,metric,title,ylabel,log in chart_specs:
        lines += ["",f"![{title}]({stem}.png)"]
    lines += ["","## 分角色工作", "", "各列独立取10轮中位数，因此中位数列的相加不必恰好等于系统工作中位数。客户端整列是全体客户端工作；VLMRS无边缘角色，其0值代表不适用。", "",
        "| N | 维度 | 实现 | 全部客户端/s | 全部边缘/s | TA/s | 云/s |", "|---:|---:|---|---:|---:|---:|---:|"]
    for r in records:
        a=r["metrics"]
        lines.append(f"| {r['N']} | {r['d']} | {CN[r['scheme']]} | {a['user_work_s']['median']:.4f} | {a['edge_work_s']['median']:.4f} | {a['ta_work_s']['median']:.4f} | {a['cloud_work_s']['median']:.4f} |")
    lines += ["","## 本矩阵中的成本比值", "", "比值为HVSA中位数/该基线中位数，小于1表示HVSA该项成本较低，大于1表示较高。所有方向如实保留；这是本实现、本配置的比值，不是安全等价或全领域排名。", "",
        "| N | 维度 | 对照 | 客户端比值 | 系统工作比值 | 全网通信比值 |", "|---:|---:|---|---:|---:|---:|"]
    for n,m,d in POINTS:
        own=next(r for r in records if r["N"]==n and r["d"]==d and r["scheme"]==ORDER[0])["metrics"]
        for scheme in ORDER[1:]:
            base=next(r for r in records if r["N"]==n and r["d"]==d and r["scheme"]==scheme)["metrics"]
            ratio=[own[k]["median"]/base[k]["median"] for k in ["client_ms_per_selected","system_work_s","traffic_mib"]]
            lines.append(f"| {n} | {d} | {CN[scheme]} | {ratio[0]:.6g} | {ratio[1]:.4f} | {ratio[2]:.4f} |")
    lines += ["","## 初始化单列", "", "每点每方案各初始化一次，下表不是重复统计分布。GPVO公共RSA/LHF参数生成计于初始化，其逐轮DH/承诺随机量更新与分发已计入每轮TA/客户端/消息成本。不同点生成不同RSA公共参数，原始数值和位数在manifest中保留。", "",
        "| N | 维度 | 实现 | 初始化总工作/s | 初始化封装字节 |", "|---:|---:|---|---:|---:|"]
    for item in initialization:
        lines.append(f"| {item['N']} | {item['d']} | {CN[item['scheme']]} | {item['total_work_s']:.4f} | {item['sent_bytes']} |")
    lines += ["","## 实例化与结论边界", "",
        "GPVO为Z*_(N²)哈希、3072位模数、素数e>N²、每轮新会话的明确实例化；随机单位元g未提供最大阶证书，也不声称其整数群接口等同原文P-256实现。模幂使用gmpy2原生后端。VLMRS采用全邻居、ceil(2N/3)阈值、公开根指数7、共享秘密线性HF及长期DH密钥复用；配对使用py-ecc 8.0.0。HVSA使用当前真实根实体。", "",
        "信任设施、隐私/认证性质和密码库后端存在差异。HVSA生成元缓存容量4096，GPVO为131072；二者仅在各状态初始化时清空，后续不逐轮清空。7850维时缓存行为不同，不能将实现成本差都归因于协议结构。GPVO/VLMRS已有条件错误接受和历史恢复现象继续保留；诚实回合全部正确不能改写为安全等价。本矩阵没有完成生产参数的完整E5矩阵、跨方案50轮/掉线矩阵或额外安全证明，不能据此宣称SOTA。通信为实际逐接收者协议封装，不包含TCP/TLS头或估计RTT。", "",
        "本阶段交付了四点重复成本观测及三类图。后续按实际论文需要补生产参数异常行为、同人数多轮和分阶段掉线；既有HVSA消融与训练结果另列，不重复计数。", "",
        "## 可追溯来源", "", f"批次：`{batch_path}`，SHA-256 `{digest(batch_path)}`。", ""]
    for job in journal["jobs"]:
        lines.append(f"- `{job['run_dir']}`，manifest SHA-256 `{job['manifest_sha256']}`。")
    lines += ["", "analysis.json保存逐次成本、统计定义、参数、源码/输出摘要和来源。PNG/SVG为本次实际数据导出；图形生成后另做目视核查。", ""]
    (output/"RESULTS_SUMMARY.md").write_text("\n".join(lines),encoding="utf-8")
    shutil.copy2(__file__,output/"analysis_source.py")
    dump_json(output/"analysis_manifest.json",{"artifact_kind":"analysis_of_existing_evidence",
        "created_at":datetime.datetime.now().astimezone().isoformat(),"visual_review":"pending",
        "batch":{"path":str(batch_path),"sha256":digest(batch_path)},"matplotlib":matplotlib.__version__,"numpy":np.__version__,
        "output_sha256":{str(p.relative_to(output)):digest(p) for p in output.rglob("*") if p.is_file()}})
    print("ANALYSIS_COMPLETED="+str(output),flush=True)


if __name__=="__main__": main()
