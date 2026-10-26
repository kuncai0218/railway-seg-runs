"""组长端（10-26，Claude）：从 ntfy.sh 抓三位同学的状态消息和附件，存进本仓库并推送到 GitHub 做存档。
在仓库根目录运行：python tools/collect.py            （抓一次；本机 Codex 每 30 分钟运行一次）
          python tools/collect.py --no-push  （只存本地，不推送）
ntfy.sh 上的消息只保留约 12 小时、附件约 3 小时，所以要定时抓。"""
import argparse
import datetime
import json
import subprocess
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUFFIX = 'd4a86d'
PEOPLE = {'jia': '同学甲', 'yi': '同学乙', 'bing': '同学丙'}


def poll(topic):
    raw = urllib.request.urlopen(f'https://ntfy.sh/{topic}/json?poll=1&since=all', timeout=60).read().decode('utf-8')
    return [json.loads(l) for l in raw.splitlines() if l.strip() and json.loads(l).get('event') == 'message']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--no-push', action='store_true'); a = ap.parse_args()
    rows, changed = [], 0
    for slug, name in PEOPLE.items():
        d = REPO / name; d.mkdir(exist_ok=True)
        seen_f = d / '.seen.json'
        seen = set(json.loads(seen_f.read_text('utf-8'))) if seen_f.exists() else set()
        try:
            msgs = poll(f'railseg-{SUFFIX}-status-{slug}')
        except Exception as e:   # noqa: BLE001
            print(name, '抓取失败', e); continue
        for m in msgs:
            if m['id'] in seen:
                continue
            t = datetime.datetime.fromtimestamp(m['time'])
            rec = {'id': m['id'], 'time': t.strftime('%Y-%m-%d %H:%M:%S'), 'message': m.get('message', '')}
            att = m.get('attachment')
            if att:
                try:
                    body = urllib.request.urlopen(att['url'], timeout=60).read()
                    (d / att['name']).write_bytes(body); rec['attachment'] = att['name']
                    (d / '最新状态.json').write_bytes(body)
                except Exception as e:   # noqa: BLE001  附件过期（约 3 小时）就只留消息
                    rec['attachment_error'] = str(e)
            with open(d / '消息记录.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            seen.add(m['id']); changed += 1
        seen_f.write_text(json.dumps(sorted(seen)), encoding='utf-8')
        last = (d / '消息记录.jsonl').read_text('utf-8').splitlines()[-1] if (d / '消息记录.jsonl').exists() else None
        rows.append((name, json.loads(last) if last else None))
    lines = ['# 各同学最新状态', '', f'更新于 {datetime.datetime.now():%Y-%m-%d %H:%M}', '', '| 同学 | 时间 | 摘要 |', '|---|---|---|']
    for name, r in rows:
        lines.append(f"| {name} | {r['time'] if r else '—'} | {r['message'].replace('|', '／') if r else '还没有消息'} |")
    (REPO / '最新状态.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'新消息 {changed} 条')
    if changed and not a.no_push:
        subprocess.run(['git', '-C', str(REPO), 'add', '-A'], check=True)
        subprocess.run(['git', '-C', str(REPO), 'commit', '-q', '-m', f'状态存档 {datetime.datetime.now():%m-%d %H:%M}'], check=True)
        subprocess.run(['git', '-C', str(REPO), 'push', '-q'], check=True)
        print('已推送到 GitHub')


if __name__ == '__main__':
    main()
