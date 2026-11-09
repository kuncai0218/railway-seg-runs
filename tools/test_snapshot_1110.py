"""Regression cases for the two incompatible current reporting formats."""
from snapshot_1110 import normalize_current

def main():
 a=normalize_current({'instruction_id':'jia-st3d-1109','tasks':{'muds_gn_s42':{'completed_epochs':58},'muds_gn_s43':{'completed_epochs':58}},'time':'now'})
 assert a['state']=='running' and len(a['reported_active_tasks'])==2 and a['completed_parts']==0
 d={'指令编号':'yi-st3d-1109','任务进度':{'den_pseg0_3src_s43':{'已完成轮数':16},'den_pseg0_3src_s44':{'已完成轮数':16},'den3src_s43':{'已完成轮数':55}},'运行器日志尾部':['11-10 03:10:00 muds_paper_s44 已完成，跳过','运行器退出']}
 for arch in ['gn','pseg0']:
  for seed in [43,44]:d['任务进度'][f'den_{arch}_planet_s{seed}']={'已完成轮数':30,'final.json':True}
 b=normalize_current(d)
 assert b['completed_parts']==4 and len(b['reported_active_tasks'])==2 and b['state']=='running' and 'den3src_s43' not in b['task_progress']
 assert normalize_current({'parts':{}}) is None
 d['任务进度']['den_gn_3src_s43']={'已完成轮数':3,'final.json':True}
 assert normalize_current(d)['completed_parts']==4,'Incomplete finals are not accepted'
 d['指令编号']='yi-st3d-1109；已收到接管，后续yi-st3d-1110-v2待执行'
 assert normalize_current(d)['completed_parts']==4,'Future batch mentioned in acknowledgment must not override active batch'
 d['进程']={'训练进程':[{'任务':'den_gn_3src_s43','pid':123}]};d['任务进度']['den_gn_3src_s43']={'已完成轮数':0}
 assert 'den_gn_3src_s43' in normalize_current(d)['reported_active_tasks'],'Newly started first epoch must not appear idle'
 d['进程']={'训练进程':[{'任务':'den_gn_3src_v2_1110_s43','pid':124}]}
 assert normalize_current(d)['total_parts']==4,'Live new batch takes precedence over stale instruction text'
 d.pop('进程')
 d['指令编号']='yi-st3d-1110-v2';d['任务进度']['den_pseg0_3src_v2_1110_s43']={'已完成轮数':1}
 c=normalize_current(d)
 assert c['completed_parts']==0 and c['reported_active_tasks']==['den_pseg0_3src_v2_1110_s43']
 print('8 normalization cases passed; history, future mentions, premature finals, first-epoch startup and stale instruction text checked.')
if __name__=='__main__':main()
