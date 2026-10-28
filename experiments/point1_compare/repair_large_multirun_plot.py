"""Preserve the original render and fix wire curves hidden against upper spines."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
from .artifacts import dump_json
from .round_shards import file_sha
from .large_multirun_contract import SCHEMES,GPVO,VLMRS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output')
    args=p.parse_args();out=Path(args.output).resolve()
    data=json.loads((out/'analysis.json').read_text(encoding='utf-8'))
    manifest=json.loads((out/'analysis_manifest.json').read_text(encoding='utf-8'))
    assert manifest['visual_review']=='pending_actual_png_views'
    for name in ['analysis.json','roundwise_work_wire.png','roundwise_work_wire.svg']:
        assert file_sha(out/name)==manifest['outputs'][name]
    history=out/'review_history/initial_roundwise_plot';history.mkdir(parents=True,exist_ok=False)
    for name in ['analysis_manifest.json','roundwise_work_wire.png','roundwise_work_wire.svg']:
        shutil.copyfile(out/name,history/name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    curves=data['curves'];colors={'hvsa':'#2474a6',GPVO:'#de8a25',VLMRS:'#3b9866'}
    names={'hvsa':'HVSA',GPVO:'GPVO declared',VLMRS:'VLMRS declared'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True)
    max_work=max(max(c['round_work_s']) for c in curves)
    max_wire=max(max(c['round_wire_bytes'])/2**20 for c in curves)
    for row,scheme in enumerate(SCHEMES):
        for label,style in [('fixed','-'),('changing','--')]:
            c=next(v for v in curves if v['scheme']==scheme and v['label']==label)
            for col,y in [(0,c['round_work_s']),(1,np.array(c['round_wire_bytes'])/2**20)]:
                axes[row,col].plot(c['rounds'],y,style,color=colors[scheme],linewidth=1.4,
                    marker='o' if label=='fixed' else 's',markersize=2,
                    markerfacecolor=colors[scheme] if label=='fixed' else 'white',label=label)
                failed=[i for i,v in enumerate(c['valid']) if not v]
                if failed:axes[row,col].scatter(np.array(c['rounds'])[failed],np.array(y)[failed],marker='x',color='red',zorder=4)
        for col in range(2):
            ax=axes[row,col];ax.set_title(names[scheme]+(' / round work' if col==0 else ' / wire envelopes'))
            ax.set_ylabel('All-role work (s)' if col==0 else 'Protocol envelopes (MiB)')
            ax.set_xlim(.5,50.5);ax.set_ylim(0,1.1*(max_work if col==0 else max_wire))
            ax.grid(alpha=.2);ax.legend(frameon=False,ncol=2,fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('Actual round index + 1')
    fig.text(.5,.035,'Every observed round marked; shared y limits within each column. First-round effects and outliers retained; setup excluded.',ha='center',fontsize=8.5)
    fig.text(.5,.015,'50 registered, 40 uploaded, 7850 dimensions; local sequential replay. Red x (if any) marks incomplete/invalid returned calls.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.085,right=.98,bottom=.1,top=.95,hspace=.36,wspace=.23)
    fig.savefig(out/'roundwise_work_wire.png',dpi=200);fig.savefig(out/'roundwise_work_wire.svg');plt.close(fig)
    shutil.copyfile(__file__,out/'roundwise_plot_repair_source.py')
    shutil.copyfile(Path(__file__).with_name('report_large_multirun.py'),out/'report_large_multirun_v2.py')
    for name in ['roundwise_work_wire.png','roundwise_work_wire.svg','roundwise_plot_repair_source.py','report_large_multirun_v2.py']:
        manifest['outputs'][name]=file_sha(out/name)
    manifest['figure_repair']=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reason='Near-constant wire curves were pressed against the upper spine by automatic limits. Explicit shared column limits add 10 percent headroom; all points now marked.',
        original_manifest_sha256=file_sha(history/'analysis_manifest.json'),data_changed=False,
        changed_figures=['roundwise_work_wire.png','roundwise_work_wire.svg'])
    dump_json(out/'analysis_manifest.json',manifest)
    print('ROUNDWISE_PLOT_REPAIRED_DATA_UNCHANGED')


if __name__=='__main__':main()
