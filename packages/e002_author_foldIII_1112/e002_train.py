"""Single-GPU author MultiUTAE reproduction on DEN spatial fold II.

Pinned author network/loss/transforms; explicit paper budget and SCS selection.
No original official validation/test or internal fold test loader is created.
"""
from pathlib import Path
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[name]='2'
import argparse
import datetime as dt
import hashlib
import json
import random
import time
import numpy as np
import torch

from e002_data import WORK,VENDOR,CachedAuthorDEN,collate_fn,verify_against_native
from e002_metrics import SCDMetric
from models.networks.multiutae import MultiUTAE
from models.losses import Losses

OWN=WORK.parent
PROTOCOL=WORK/'protocol.json'


def now():return dt.datetime.now().astimezone().isoformat()


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(8*2**20):h.update(b)
    return h.hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temporary,path)


def event(out,stage,**details):
    obj={'time':now(),'PID':os.getpid(),'stage':stage,**details}
    save(out/'progress.json',obj)
    print(json.dumps(obj,ensure_ascii=False),flush=True)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def check_vendor():
    manifest=json.loads((WORK/'vendor_source_manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['tracked_file_hashes'].items():
        assert sha(VENDOR/name)==digest,name
    return manifest


def build(proto):
    net=MultiUTAE(**proto['network']).cuda()
    assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in net.modules())
    loss=Losses(**proto['loss']).cuda()
    assert loss.loss['focal'][0].gamma==0
    kwargs=dict(proto['optimizer']['optim']);kwargs.pop('_target_')
    # PyYAML retains author spelling 1e-4 as a string; Hydra coerces it.
    for key in ['lr','eps','weight_decay']:
        if key in kwargs:kwargs[key]=float(kwargs[key])
    if 'betas' in kwargs:kwargs['betas']=tuple(float(x) for x in kwargs['betas'])
    optim=torch.optim.AdamW(net.parameters(),**kwargs)
    scaler=torch.amp.GradScaler('cuda')
    return net,loss,optim,scaler


def to_cuda(samples):
    batch=collate_fn(samples)
    return {k:v.cuda(non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def verify_loss_reduction(samples):
    # Match the original single-device batch8 valid-pixel mean, including an
    # empty microbatch. Averaging microbatch means would overweight clouds.
    targets=torch.stack([s['gt'][:,::32,::32] for s in samples]);targets[:2]=6
    logits=torch.randn(8,12,6,4,4,requires_grad=True)
    fn=Losses(mix={'focal':1.},ignore_index=6)
    reference=fn({'logits':logits},{'gt':targets})['loss']
    total=int((targets!=6).sum());parts=[]
    for start in range(0,8,2):
        gt=targets[start:start+2];x=logits[start:start+2];valid=int((gt!=6).sum())
        parts.append(fn({'logits':x},{'gt':gt})['loss']*valid/total if valid else x.sum()*0)
    accumulated=sum(parts)
    g1=torch.autograd.grad(reference,logits,retain_graph=True)[0]
    g2=torch.autograd.grad(accumulated,logits)[0]
    error=float((g1-g2).abs().max());difference=abs(float(reference.detach()-accumulated.detach()))
    assert difference<1e-6 and error<1e-7
    return {'same_as_author_global_batch8_valid_pixel_mean':True,
            'empty_microbatch_case_checked':True,'loss_abs_difference':difference,'gradient_max_abs_difference':error}


def attention_hook(store):
    def hook(_module,_args,result):
        if not store.get('capture'):return
        a=result[1].detach()
        store['value']={'shape':list(a.shape),'post_dropout':True,
            'finite':bool(torch.isfinite(a).all()),'max':float(a.max()),
            'mean_query_max':float(a.amax(3).mean()),
            'mean_row_sum':float(a.sum(3).mean())}
    return hook


def admission(proto):
    out=WORK/'admission_0815';out.mkdir(exist_ok=True)
    event(out,'load_training_labels',regions=33)
    train=CachedAuthorDEN(proto['train_regions'],'train')
    comparison=verify_against_native(train)
    save(out/'CPU_exact_data_equivalence.json',{'time':now(),'passed':True,
        'cases':comparison,'reference':'pinned author full-frame JPEG decode -> float16 -> exact source transforms/normalization',
        'heldout_validation_or_test_arrays_read':False})
    event(out,'CPU_exact_data_equivalence_passed',cases=len(comparison))
    while torch.cuda.mem_get_info()[0]<9*2**30:
        event(out,'waiting_for_GPU_headroom',free_bytes=torch.cuda.mem_get_info()[0]);time.sleep(5)
    seed_all(proto['seed']);net,loss_fn,optim,scaler=build(proto)
    net.train();capture={'capture':True};net.temporal_encoder.register_forward_hook(attention_hook(capture))
    ids=np.linspace(0,len(train)-1,8,dtype=int).tolist()
    # Admission is discarded. Use a nonzero LR to verify a genuine parameter
    # update, while the formal run below starts its paper warmup at LR zero.
    samples=[train[idx] for idx in ids]
    reduction_check=verify_loss_reduction(samples)
    total_valid=sum(int((s['gt']!=6).sum()) for s in samples)
    assert total_valid>0
    before=next(net.parameters()).detach().clone()
    torch.cuda.reset_peak_memory_stats();start=time.monotonic();optim.zero_grad(set_to_none=True)
    losses=[]
    for sample in samples:
        batch=to_cuda([sample]);count=int((sample['gt']!=6).sum())
        with torch.autocast('cuda',dtype=torch.float16):
            output=net(batch)
            loss=loss_fn(output,batch,average=True)['loss']*count/total_valid if count else output['logits'].float().sum()*0
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();losses.append(float(loss.detach()))
    scaler.unscale_(optim)
    gradients=[p.grad for p in net.parameters() if p.grad is not None]
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
    scaler.step(optim);scaler.update();torch.cuda.synchronize()
    assert not torch.equal(before,next(net.parameters()).detach())
    elapsed=time.monotonic()-start
    result={'time':now(),'passed':True,'CPU_exact_data_equivalence_passed':True,
        'optimizer_updates':1,'formal_training_started':False,'weights_discarded_after_admission':True,
        'global_samples':8,'micro_batch_size':1,'accumulation':8,'train_length':12,'channels':4,
        'weighted_microbatch_loss_terms':losses,'loss_reduction_check':reduction_check,'finite_gradients':True,
        'seconds_for_one_effective_batch8_update':elapsed,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'attention':capture.get('value'),'parameter_count':sum(p.numel() for p in net.parameters()),
        'protocol_SHA256':sha(PROTOCOL),'official_commit':proto['author_commit'],
        'environment':{'torch':torch.__version__,'CUDA':torch.version.cuda,
            'GPU':torch.cuda.get_device_name(),'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'allow_tf32_matmul':torch.backends.cuda.matmul.allow_tf32,
            'allow_tf32_cudnn':torch.backends.cudnn.allow_tf32},
        'script_SHA256':sha(Path(__file__)),'data_adapter_SHA256':sha(WORK/'e002_data.py')}
    save(WORK/'admission_accepted.json',result);event(out,'admission_completed',**result)
    with (OWN/'R03_远端进度.md').open('a',encoding='utf-8') as f:
        f.write(f'- {dt.datetime.now():%Y-%m-%d %H:%M} E002本机准入完成：固定作者原模型/损失，DEN4通道随机12期的3个真实原JPEG与SSD裁块样本逐值一致；有效batch8的一次GPU优化器更新、有限梯度及注意力/显存检查通过，准入权重丢弃。原论文空间折II正式训练单独从头开始，见T03_files/E002_local_author_1112/admission_accepted.json。\n')


@torch.no_grad()
def evaluate(net,dataset,out,updates):
    net.eval();metric=SCDMetric(6,['impervi','agricult','forest','wetlands','soil','water'],ignore_index=6)
    start=time.monotonic()
    for index in range(len(dataset)):
        batch=to_cuda([dataset[index]])
        with torch.autocast('cuda',dtype=torch.float16):logits=net(batch)['logits']
        if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite validation logits before argmax')
        pred=logits.argmax(2)
        metric.update(pred,batch['gt'])
        if index==0 or (index+1)%64==0:
            event(out,'validation',optimizer_updates=updates,completed_patches=index+1,total_patches=len(dataset))
    values={k:float(v) for k,v in metric.compute().items()};assert all(np.isfinite(v) for v in values.values())
    net.train();return values,(time.monotonic()-start)/60


def train_model(proto,out,micro):
    admission_record=json.loads((WORK/'admission_accepted.json').read_text(encoding='utf-8'))
    assert admission_record['passed'] and admission_record['protocol_SHA256']==sha(PROTOCOL)
    assert admission_record['script_SHA256']==sha(Path(__file__))
    assert admission_record['data_adapter_SHA256']==sha(WORK/'e002_data.py')
    assert micro in [1,2] and 8%micro==0
    out.mkdir(parents=True,exist_ok=True);event(out,'loading_author_foldIII',micro_batch=micro,effective_batch=8)
    train=CachedAuthorDEN(proto['train_regions'],'train')
    val=CachedAuthorDEN(proto['validation_regions'],'val')
    assert not set(train.sits_ids)&set(val.sits_ids)
    assert not (set(train.sits_ids)|set(val.sits_ids))&set(proto['internal_test_regions_locked'])
    seed_all(proto['seed']);net,loss_fn,optim,scaler=build(proto)
    capture={'capture':False};net.temporal_encoder.register_forward_hook(attention_hook(capture))
    generator=torch.Generator().manual_seed(proto['seed'])
    permutation=torch.randperm(len(train),generator=generator);cursor=0;updates=0;attempts=0;epoch=0;best=-float('inf')
    source_hashes={p.name:sha(p) for p in [Path(__file__),WORK/'e002_data.py',WORK/'e002_metrics.py',PROTOCOL]}
    runconfig={'protocol':proto,'source_hashes':source_hashes,'official_source_manifest':str(WORK/'vendor_source_manifest.json'),
        'micro_batch':micro,'accumulation':8//micro,'global_batch':8,'validation_regions':val.sits_ids,
        'internal_test_loader_created':False,'original_official_val_test_loader_created':False,
        'loss_reduction':'original single-device global batch8 valid-pixel mean; each microbatch weighted by valid-pixel count. Original multi-GPU rank reduction is not claimed byte-identical.',
        'sampler':'whole epoch random permutation; global batch8 drop_last; author monthly lists shuffled in main process',
        'checkpoint_RNG_scope':'Python/NumPy/Torch CPU and CUDA, batch generator/permutation/cursor, mutable author month lists',
        'floating_point_note':'RNG and sampling restored on resume; CuDNN nondeterminism is not claimed bit-exact'}
    configpath=out/'config.json'
    if configpath.exists():assert json.loads(configpath.read_text(encoding='utf-8'))==runconfig
    else:save(configpath,runconfig)
    checkpoint=out/'last.pth'
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['source_hashes']==source_hashes
        net.load_state_dict(state['model']);optim.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        updates=state['optimizer_updates'];attempts=state['attempts'];epoch=state['completed_epochs'];best=state['best_validation_scs']
        permutation=state['permutation'];cursor=state['cursor'];generator.set_state(state['generator_state'])
        train.restore_sampling_state(state['month_lists'])
        random.setstate(state['Python_RNG']);np.random.set_state(state['NumPy_RNG']);torch.set_rng_state(state['Torch_RNG'])
        torch.cuda.set_rng_state_all(state['CUDA_RNG']);event(out,'resumed',optimizer_updates=updates,completed_epochs=epoch)

    def checkpoint_state():
        return {'model':net.state_dict(),'optimizer':optim.state_dict(),'scaler':scaler.state_dict(),
            'optimizer_updates':updates,'attempts':attempts,'completed_epochs':epoch,'best_validation_scs':best,
            'permutation':permutation,'cursor':cursor,'generator_state':generator.get_state(),
            'month_lists':train.sampling_state(),'Python_RNG':random.getstate(),'NumPy_RNG':np.random.get_state(),
            'Torch_RNG':torch.get_rng_state(),'CUDA_RNG':torch.cuda.get_rng_state_all(),
            'source_hashes':source_hashes,'protocol_SHA256':sha(PROTOCOL),'time':now()}

    last_checkpoint_time=time.monotonic()

    def save_checkpoint(archive=False):
        nonlocal last_checkpoint_time
        value=checkpoint_state();temporary=out/'last.pth.tmp';torch.save(value,temporary);os.replace(temporary,checkpoint)
        if archive:torch.save(value,out/f'step_{updates:07d}.pth')
        last_checkpoint_time=time.monotonic()

    net.train();started=time.monotonic();start_updates=updates;epoch_loss=[]
    values=None
    # A completed checkpoint may survive an interruption before final.json.
    # Recover the already-frozen source validation result, without a test read.
    if updates>=proto['max_optimizer_updates']:
        log=out/'validation.jsonl'
        recorded=[json.loads(s) for s in log.read_text(encoding='utf-8').splitlines()] if log.exists() else []
        match=[r for r in recorded if r['optimizer_updates']==updates]
        values=match[-1]['metrics'] if match else evaluate(net,val,out,updates)[0]
    while updates<proto['max_optimizer_updates']:
        assert cursor+8<=len(permutation)
        ids=permutation[cursor:cursor+8].tolist();cursor+=8;attempts+=1
        lr=proto['learning_rate']*min(updates/proto['warmup_optimizer_updates'],1.)
        for group in optim.param_groups:group['lr']=lr
        optim.zero_grad(set_to_none=True);total_loss=0.;capture['capture']=updates==0 or (updates+1)%100==0
        if capture['capture']:capture['sample_update']=updates+1
        scale_before=scaler.get_scale()
        samples=[train[index] for index in ids]
        total_valid=sum(int((s['gt']!=6).sum()) for s in samples)
        for i in range(0,8,micro):
            selected=samples[i:i+micro];batch=to_cuda(selected)
            count=sum(int((s['gt']!=6).sum()) for s in selected)
            with torch.autocast('cuda',dtype=torch.float16):
                output=net(batch)
                loss=loss_fn(output,batch,average=True)['loss']*count/total_valid if count else output['logits'].float().sum()*0
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite author focal loss; preserve checkpoint and diagnose')
            scaler.scale(loss).backward();total_loss+=float(loss.detach())
        if total_valid:
            scaler.step(optim);scaler.update()
        else:
            scaler.unscale_(optim);scaler.update()
        if not total_valid:
            event(out,'all_ignored_global_batch_skipped',optimizer_updates=updates,attempts=attempts)
        elif scaler.get_scale()<scale_before:
            event(out,'AMP_overflow_optimizer_skipped',optimizer_updates=updates,attempts=attempts,scale=scaler.get_scale())
        else:
            updates+=1;epoch_loss.append(total_loss)
            if updates==1 or updates%proto.get('progress_every_optimizer_updates',10)==0:
                elapsed=time.monotonic()-started
                event(out,'training',optimizer_updates=updates,attempts=attempts,completed_epochs=epoch,
                    loss=total_loss,learning_rate=lr,AMP_scale=scaler.get_scale(),micro_batch=micro,effective_batch=8,
                    elapsed_seconds=elapsed,seconds_per_optimizer_update=elapsed/max(updates-start_updates,1),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),attention=capture.get('value'),
                    attention_sample_optimizer_update=capture.get('sample_update'))
            if updates==100:
                save_checkpoint(archive=True)
                save(out/'first100_updates.json',{'time':now(),'actual_optimizer_updates':100,'attempts':attempts,
                    'seconds':time.monotonic()-started,'micro_batch':micro,'effective_batch':8,'from_scratch':True,
                    'source_hashes':source_hashes,'continues_to_paper_budget':True})
        completed_epoch=cursor+8>len(permutation)
        if completed_epoch:
            epoch+=1;cursor=0;permutation=torch.randperm(len(train),generator=generator)
            with (out/'epochs.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps({'time':now(),'completed_epoch':epoch,'optimizer_updates':updates,
                                    'mean_loss':float(np.mean(epoch_loss)) if epoch_loss else None})+'\n')
            epoch_loss=[]
        due_validation=completed_epoch and epoch%proto['validation_every_completed_epochs']==0
        final=updates>=proto['max_optimizer_updates']
        if due_validation or final:
            values,minutes=evaluate(net,val,out,updates)
            with (out/'validation.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'time':now(),'optimizer_updates':updates,'epoch':epoch,'metrics':values,'minutes':minutes})+'\n')
            if values['scs']>best:
                best=values['scs'];torch.save({'model':net.state_dict(),'optimizer_updates':updates,'metrics':values,'source_hashes':source_hashes},out/'best_scs.pth')
            save_checkpoint(archive=final or updates%proto['archive_every_optimizer_updates']==0)
        elif updates and (updates%proto['checkpoint_every_optimizer_updates']==0 or
                          time.monotonic()-last_checkpoint_time>=proto.get('checkpoint_wall_seconds',600)):
            save_checkpoint(archive=updates%proto['archive_every_optimizer_updates']==0)
    save(out/'final.json',{'time':now(),'optimizer_updates':updates,'attempts':attempts,'completed_epochs':epoch,
        'best_validation_scs':best,'last_validation':values,'internal_test_evaluated':False,'source_hashes':source_hashes})
    event(out,'completed',optimizer_updates=updates,completed_epochs=epoch,best_validation_scs=best)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--admission',action='store_true')
    parser.add_argument('--micro',type=int,default=2);parser.add_argument('--out',type=Path,default=OWN/'runs/E002_DEN_author_spatial_III_s42_1112')
    args=parser.parse_args();torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    proto=json.loads(PROTOCOL.read_text(encoding='utf-8'));check_vendor()
    if args.admission:admission(proto)
    else:train_model(proto,args.out,args.micro)


if __name__=='__main__':main()
