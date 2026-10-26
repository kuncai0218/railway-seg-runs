"""同学端状态同步（10-26，Claude）：把实验进度和精度发到 ntfy.sh 公开话题，并读取组长的批注。不需要任何账号。

放在实验包的 code/ 目录里，在实验包根目录运行：
    python code/sync_status.py --name 同学甲 --task tasks/同学甲.json            # 同步一次
    python code/sync_status.py --name 同学甲 --task tasks/同学甲.json --loop 3600  # 每小时同步一次，一直运行
每次同步：
  1. 发一条状态消息（一句话摘要）+ 一个附件 status_<名字>_<时间>.json（进度表、各部分精度、学习率选择、训练曲线、日志末尾、出错信息）；
  2. 读取批注话题里的新批注，追加到 results/批注_收到.md，并在屏幕上打印出来。
批注只是文字：只有第 5 节“允许的指令”才由 agent 按固定步骤执行，其他内容只转告同学本人。
只用 Python 标准库，Windows / Linux 都能用。"""
import argparse
import datetime
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUFFIX = 'd4a86d'
SLUG = {'同学甲': 'jia', '同学乙': 'yi', '同学丙': 'bing'}
MAX_ATTACH = 12 * 2**20


def now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def tail(path, n=40):
    try:
        return path.read_text('utf-8', errors='replace').splitlines()[-n:]
    except OSError:
        return []


def collect(name, task, res):
    st = {'name': name, 'time': now(), 'task_file': task.name}
    try:
        st['status_table'] = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'code' / 'run_tasks.py'), '--task', str(task), '--out', str(res), '--status'],
                                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120).stdout
    except Exception as e:      # noqa: BLE001
        st['status_table'] = f'读取进度失败：{e}'
    st['runner_log_tail'] = tail(res / 'runner_log.txt', 40)
    st['self_check'] = json.loads((res / '自检报告.json').read_text('utf-8')) if (res / '自检报告.json').exists() else None
    st['chosen_lr'] = {f.stem: json.loads(f.read_text('utf-8')) for f in sorted(res.glob('chosen_lr_*.json'))}
    parts, curves, failures = {}, {}, {}
    for run in sorted(res.glob('pred_*')):
        for c in sorted(run.glob('*/state_*/complete.json')):
            d = json.loads(c.read_text('utf-8'))
            parts[f'{run.name}/{c.parent.parent.name}/{c.parent.name}'] = {k: d.get(k) for k in ('mIoU', 'OA', 'IoU', 'seconds_this_session', 'peak_gpu_gb', 'time')}
        for c in sorted(run.glob('*/state_*/curve.json')):
            curves[f'{run.name}/{c.parent.parent.name}/{c.parent.name}'] = [(p['step'], round(p['mIoU'], 4)) for p in json.loads(c.read_text('utf-8'))]
        if (run / 'failure.json').exists():
            failures[run.name] = json.loads((run / 'failure.json').read_text('utf-8')).get('error')
        if (run / 'evaluation.json').exists():
            ev = json.loads((run / 'evaluation.json').read_text('utf-8'))
            st.setdefault('evaluation', {})[run.name] = {f: {k: (v.get(k) if isinstance(v, dict) else v) for k in ('development_crossfit', 'test')}
                                                       for f, v in ev.get('completed_folds', {}).items()}
    st['parts'], st['curves'], st['failures'] = parts, curves, failures
    bad = [l for l in st['runner_log_tail'] if '出错' in l]
    st['error_logs'] = {}
    for line in bad[-3:]:
        nm = line.split('结束 ')[-1].split('，')[0]
        st['error_logs'][nm] = tail(res / 'logs' / f'{nm}.log', 30)
    done = len(parts)
    last = max(parts.items(), key=lambda kv: kv[1].get('time') or '', default=None)
    summary = f"{name} | 已完成 {done} 个部分 | 出错 {len(bad)} 处"
    if last:
        summary += f" | 最新 {last[0].replace('pred_', '')} mIoU {last[1]['mIoU']:.3f}"
    if bad:
        summary += f" | 最近出错：{bad[-1][-60:]}"
    return st, summary


def publish(topic, message, title, attach_bytes, filename):
    q = urllib.parse.urlencode({'message': message[:3500], 'title': title, 'filename': filename})
    req = urllib.request.Request(f'https://ntfy.sh/{topic}?{q}', data=attach_bytes, method='PUT')
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def read_notes(topic, res):
    seen_f = res / '.notes_seen.json'
    seen = set(json.loads(seen_f.read_text('utf-8'))) if seen_f.exists() else set()
    raw = urllib.request.urlopen(f'https://ntfy.sh/{topic}/json?poll=1&since=all', timeout=60).read().decode('utf-8')
    new = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        if m.get('event') == 'message' and m['id'] not in seen:
            new.append(m); seen.add(m['id'])
    if new:
        with open(res / '批注_收到.md', 'a', encoding='utf-8') as f:
            for m in new:
                t = datetime.datetime.fromtimestamp(m['time']).strftime('%Y-%m-%d %H:%M')
                f.write(f"\n## {t}（编号 {m['id']}）\n{m.get('message', '')}\n")
        seen_f.write_text(json.dumps(sorted(seen)), encoding='utf-8')
    return new


def once(a):
    res = a.out
    res.mkdir(parents=True, exist_ok=True)
    slug = SLUG.get(a.name, a.name)
    st, summary = collect(a.name, a.task, res)
    data = json.dumps(st, ensure_ascii=False, indent=1).encode('utf-8')
    if len(data) > MAX_ATTACH:          # 超长时去掉曲线和日志，只留结果
        st.pop('curves', None); st.pop('error_logs', None)
        data = json.dumps(st, ensure_ascii=False).encode('utf-8')
    stamp = datetime.datetime.now().strftime('%m%d_%H%M')
    r = publish(f'railseg-{SUFFIX}-status-{slug}', summary, f'{a.name} 状态', data, f'status_{slug}_{stamp}.json')
    print(f'{now()} 已发送状态：{summary}（消息编号 {r.get("id")}）', flush=True)
    new = read_notes(f'railseg-{SUFFIX}-notes-{slug}', res)
    for m in new:
        print(f'{now()} 【收到组长批注】{m.get("message", "")}', flush=True)
    if not new:
        print(f'{now()} 没有新批注', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, choices=list(SLUG))
    ap.add_argument('--task', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'results')
    ap.add_argument('--loop', type=int, default=0, help='大于 0 时每隔这么多秒同步一次，一直运行')
    a = ap.parse_args()
    while True:
        try:
            once(a)
        except Exception as e:      # noqa: BLE001  网络偶尔不通：记一行，下次再试
            print(f'{now()} 同步失败（下次再试）：{e}', flush=True)
        if a.loop <= 0:
            break
        time.sleep(a.loop)


if __name__ == '__main__':
    main()
