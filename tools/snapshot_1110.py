"""Normalize the two current status schemas without mixing previous batches."""
import re

def normalize_current(snapshot):
 tasks=snapshot.get('tasks') or snapshot.get('任务进度')
 if not isinstance(tasks,dict):return None
 instruction=str(snapshot.get('instruction_id') or snapshot.get('指令编号') or '')
 live={p.get('任务') for p in snapshot.get('进程',{}).get('训练进程',[]) if isinstance(p,dict)}
 for p in snapshot.get('processes',{}).get('trainers_or_benchmarks',[]):
  argv=p.get('argv',[])
  if '--out' in argv:live.add(argv[argv.index('--out')+1].replace('\\','/').rsplit('/',1)[-1])
 if instruction.startswith('jia-st3d-1109'):
  names=['muds_gn_s42','muds_gn_s43','muds_pseg0_s42','muds_pseg0_s43'];total=250
 elif instruction.startswith('yi-st3d-1110') or any('_v2_1110_' in n for n in live if isinstance(n,str)):
  names=[f'den_{arch}_3src_v2_1110_s{seed}' for arch in ['pseg0','gn'] for seed in [43,44]];total=30
 elif instruction.startswith('yi-st3d-1109'):
  names=[f'den_{arch}_{source}_s{seed}' for source in ['planet','3src'] for arch in ['pseg0','gn'] for seed in [43,44]];total=30
 else:return None
 current={};completed=0;active=[]
 for name in names:
  d=tasks.get(name,{})
  epoch=int(d.get('completed_epochs',d.get('已完成轮数',0)) or 0)
  final=d.get('final') or d.get('final.json')
  status='completed' if final and epoch>=total else ('reported_active' if epoch>0 or name in live else 'pending')
  if status=='completed':completed+=1
  elif status=='reported_active':active.append(name)
  current[name]={'epoch':epoch,'total_epochs':total,'status':status,'epoch_line':d.get('latest_epoch_line') or d.get('最新轮次行')}
 lines=snapshot.get('queue_log_tail') or snapshot.get('运行器日志尾部') or []
 # Old queue tails are often appended after current work; progress is taken from current task entries.
 failures=[]
 for name in names:
  relevant=[x for x in lines if re.search(r'(?<!\w)'+re.escape(name)+r'(?!\w)',x)]
  if relevant and re.search(r'返回码\s+(?!0\b)-?\d+|失败',relevant[-1]):failures.append(name)
 state='failed' if failures else ('completed' if completed==len(names) else ('running' if active else 'unknown'))
 return {'source_snapshot_at':snapshot.get('time'),'state':state,'completed_parts':completed,'total_parts':len(names),'current_run_failed_tasks':failures,'historical_failure_events':sum(bool(re.search(r'返回码\s+(?!0\b)-?\d+|失败',x)) for x in lines),'latest_runner_line':lines[-1] if lines else None,'current_instruction':instruction,'task_progress':current,'reported_active_tasks':active,'evidence_note':'Reported current-batch epochs, not a claim of current GPU utilization; progression and freshness are checked separately.'}
