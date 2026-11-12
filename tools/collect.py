"""组长端（10-26，Claude）：从 ntfy.sh 抓甲、乙和本机代跑队列的状态消息和附件，存档到 GitHub。
在仓库根目录运行：python tools/collect.py            （抓一次；本机 Codex 每 30 分钟运行一次）
          python tools/collect.py --no-push  （只存本地，不推送）
ntfy.sh 上的消息只保留约 12 小时、附件约 3 小时，所以要定时抓。"""
import argparse
import datetime
import json
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUFFIX = 'd4a86d'
PEOPLE = {'jia': '同学甲', 'yi': '同学乙', 'bing': '同学丙'}
DISPLAY_NAMES = {'同学丙': '本机3090（原丙队列）'}
MAX_ATTACHMENT = 12 * 2**20


def instruction_identity(snapshot):
    value = snapshot.get('instruction_id', snapshot.get('指令编号', ''))
    if isinstance(value, dict):
        value = value.get('编号', value.get('instruction_id', ''))
    return value.lower() if isinstance(value, str) else ''


def timestamp(value):
    return datetime.datetime.fromisoformat(str(value).replace('Z', '+00:00')).timestamp()


def process_created(process):
    value = process.get('CreationDate', process.get('create_time'))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.fullmatch(r'/Date\((\d+)(?:[+-]\d{4})?\)/', value)
        if match:
            return int(match.group(1)) / 1000
        try:
            return timestamp(value)
        except ValueError:
            pass
    return None


def keep_direct_e002_receipt(folder, body):
    """Keep typed E002 evidence separately from later legacy snapshots."""
    snapshot = json.loads(body)
    identity = instruction_identity(snapshot)
    if not re.match(r'^(jia|yi)-e002-', identity):
        return None
    when = datetime.datetime.fromisoformat(str(snapshot['time']).replace('Z', '+00:00')).timestamp()
    target = folder / '最近E002专用回执.json'
    if target.exists():
        previous = json.loads(target.read_bytes())
        old = datetime.datetime.fromisoformat(str(previous['time']).replace('Z', '+00:00')).timestamp()
        if old > when:
            return None
    target.write_bytes(body)
    return target


def runtime_status(snapshot):
    """Interpret reported runner events; retained failure.json files are history."""
    pipeline = snapshot.get('E002_pipeline_current')
    identity = instruction_identity(snapshot)
    direct = bool(re.match(r'^(jia|yi)-e002-', identity))
    creation = None
    command_matches = None
    if isinstance(pipeline, dict) and pipeline.get('observed_at'):
        progress = pipeline.get('training_progress') or {}
        process = pipeline.get('trainer_process') or {}
        observed = pipeline['observed_at']
        alive = process.get('alive') and process.get('matches_recorded_creation')
        creation = process.get('os_created_epoch')
        stage = (pipeline.get('pipeline_progress') or {}).get('stage', 'unknown')
    elif direct:
        events = snapshot.get('最近事件') or []
        observed = snapshot.get('time')
        processes = snapshot.get('训练进程') or []
        candidates = events or [snapshot.get('training_progress') or snapshot.get('progress') or {}]
        progress = {}
        alive = False
        for candidate in reversed(candidates):
            for process in processes:
                created = process_created(process)
                if process.get('ProcessId', process.get('pid')) != candidate.get('PID') or created is None:
                    continue
                try:
                    if timestamp(candidate.get('time')) < created:
                        continue
                except (ValueError, TypeError):
                    continue
                command = process.get('CommandLine')
                run = snapshot.get('当前活动目录')
                matches = (('e002_train.py' in command.lower()) and
                           (not run or run.lower() in command.lower())) if isinstance(command, str) else None
                if matches is False:
                    continue
                progress, alive, creation, command_matches = candidate, True, created, matches
                break
            if alive:
                break
        stage = progress.get('stage', 'unknown')
    else:
        progress = None
    if progress is not None:
        try:
            stamp = datetime.datetime.fromisoformat(str(observed).replace('Z', '+00:00')).timestamp()
            step_time = datetime.datetime.fromisoformat(str(progress.get('time', observed)).replace('Z', '+00:00')).timestamp()
            age = datetime.datetime.now().timestamp() - stamp
            recent = -60 <= age <= 20 * 60 and abs(stamp - step_time) <= 300
        except (ValueError, TypeError):
            recent = False
        running = bool(alive and recent and progress.get('PID') and progress.get('optimizer_updates') is not None)
        return dict(source_snapshot_at=observed, state='running' if running else 'unknown',
                    completed_parts=0, current_run_failed_tasks=[], historical_failure_events=0,
                    kind='E002', stage=progress.get('stage', stage),
                    optimizer_updates=progress.get('optimizer_updates'),
                    trainer_PID=progress.get('PID'), process_identity_reported=bool(alive),
                    trainer_created_unix=creation, command_matches_reported_run=command_matches,
                    recent_progress_at_snapshot=recent,
                    note='receiver/preparation PID is not a trainer; old snapshots do not prove current survival')
    from snapshot_1110 import normalize_current
    normalized = normalize_current(snapshot)
    if normalized is not None:
        return normalized
    lines = snapshot.get('runner_log_tail') or []
    starts = [i for i, line in enumerate(lines) if ' 任务文件 ' in line]
    current = lines[starts[-1]:] if starts else lines
    context = re.search(r' 任务文件 ([^（\s]+)', current[0]) if starts else None
    mismatch = bool(snapshot.get('task_file') and context and snapshot['task_file'] != context.group(1))
    tasks = {}
    state = 'unknown'
    for line in current:
        start = re.search(r' 开始 (\S+)', line)
        end = re.search(r' 结束 ([^，]+)，返回码\s+(-?\d+)', line)
        if start:
            tasks[start.group(1)] = 'running'
            state = 'running'
        elif end:
            tasks[end.group(1)] = 'completed' if int(end.group(2)) == 0 else 'failed'
            state = 'between_tasks' if int(end.group(2)) == 0 else 'failed'
        elif '全部任务结束' in line:
            # A successful runner prints “没有出错”; that is not a failure.
            listed = re.search(r'出错的[:：]\s*\[([^\]]*)\]', line)
            failed = any(value == 'failed' for value in tasks.values()) or bool(listed and listed.group(1).strip())
            state = 'failed' if failed else 'completed'
    failures = sorted(name for name, status in tasks.items() if status == 'failed')
    history = sum(1 for line in lines if re.search(r'返回码\s+(?!0\b)-?\d+', line))
    result = {'source_snapshot_at': snapshot.get('time'), 'state': state,
            'completed_parts': len(snapshot.get('parts') or {}),
            'current_run_failed_tasks': failures, 'historical_failure_events': history,
            'latest_runner_line': current[-1] if current else None}
    if mismatch:
        # A dry run of another task can append a successful ending to this shared log.
        result.update(state='unknown', current_run_failed_tasks=[], log_context_mismatch=True,
                      reported_task_file=snapshot['task_file'], log_task_file=context.group(1))
    return result


def status_for_folder(folder):
    """Use a recent dedicated receipt when legacy telemetry targets an old run.

    A newer explicit zero-memory/zero-utilization hardware report conflicts with
    the prior live receipt and must not be hidden by that fallback.
    """
    latest = json.loads((folder / '最新状态.json').read_bytes())
    status = runtime_status(latest)
    dedicated = folder / '最近E002专用回执.json'
    if status.get('kind') == 'E002' or not dedicated.exists():
        return status
    value = json.loads(dedicated.read_bytes())
    current = runtime_status(value)
    if current.get('state') != 'running':
        return status
    hardware = latest.get('机器') or {}
    gpu = hardware.get('GPU', latest.get('GPU', ''))
    newer = timestamp(latest.get('time')) > timestamp(value.get('time'))
    explicit_idle = isinstance(gpu, str) and re.search(r'(?:^|,)\s*0(?:\.0+)?\s*MiB\b', gpu) and re.search(r'(?:^|,)\s*0(?:\.0+)?\s*%', gpu)
    if newer and explicit_idle:
        status.update(conflicting_newer_idle_hardware=True,
                      prior_dedicated_snapshot_at=value.get('time'))
        return status
    current.update(evidence_source='recent dedicated E002 receipt',
                   latest_legacy_snapshot_at=latest.get('time'),
                   latest_legacy_hardware_does_not_prove_steps=True)
    return current


def status_summary(status):
    if status.get('kind') == 'E002':
        count = status.get('optimizer_updates')
        detail = f'真实更新{count}' if count is not None else '尚无正式更新回报'
        label = '该次采样训练进程/创建身份与近期步数通过' if status['state'] == 'running' else '当前推进待新采样核实'
        return f'E002 {status.get("stage")}；{detail}；{label}'
    label = {'running': '运行中', 'between_tasks': '上一项已完成，等待后续日志',
             'completed': '队列已结束', 'failed': '当前运行有失败', 'unknown': '等待运行记录'}[status['state']]
    if status.get('log_context_mismatch'):
        label = '日志属于另一任务，运行状态待核实'
    summary = f"已完成 {status['completed_parts']} 个部分；{label}"
    summary += f"；本轮失败 {len(status['current_run_failed_tasks'])} 项"
    if status['historical_failure_events']:
        summary += f"；历史失败记录 {status['historical_failure_events']} 次"
    return summary


def read_attachment(att):
    name = att['name']
    if not isinstance(name, str) or '/' in name or '\\' in name or Path(name).name != name or not name.endswith('.json'):
        raise ValueError('附件必须是当前同学目录内的 JSON 文件名')
    url = urllib.parse.urlsplit(att['url'])
    if url.scheme != 'https' or url.hostname != 'ntfy.sh' or not url.path.startswith('/file/'):
        raise ValueError('附件地址不是 ntfy 文件地址')
    if int(att.get('size', 0)) > MAX_ATTACHMENT:
        raise ValueError('附件超过 12 MiB，只归档小型状态文件')
    with urllib.request.urlopen(att['url'], timeout=30) as response:
        body = response.read(MAX_ATTACHMENT + 1)
    if len(body) > MAX_ATTACHMENT:
        raise ValueError('附件超过 12 MiB')
    if not isinstance(json.loads(body), dict):
        raise ValueError('状态附件必须是 JSON 对象')
    return name, body


def poll(topic):
    raw = urllib.request.urlopen(f'https://ntfy.sh/{topic}/json?poll=1&since=all', timeout=60).read().decode('utf-8')
    return [json.loads(l) for l in raw.splitlines() if l.strip() and json.loads(l).get('event') == 'message']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-push', action='store_true')
    ap.add_argument('--peers', nargs='+', choices=list(PEOPLE), help='只采集指定来源；jia yi模式单独生成甲乙页面')
    a = ap.parse_args()
    selected = {slug: PEOPLE[slug] for slug in (a.peers or PEOPLE)}
    all_sources = set(selected) == set(PEOPLE)
    page = REPO / ('最新状态.md' if all_sources else '甲乙最新状态.md')
    rows, changed = [], 0
    written = set()
    for slug, name in selected.items():
        d = REPO / name; d.mkdir(exist_ok=True)
        seen_f = d / '.seen.json'
        seen = set(json.loads(seen_f.read_text('utf-8'))) if seen_f.exists() else set()
        try:
            msgs = poll(f'railseg-{SUFFIX}-status-{slug}')
        except Exception as e:   # noqa: BLE001
            print(name, '抓取失败，保留已有快照', e)
            msgs = []
        for m in msgs:
            if m['id'] in seen:
                continue
            t = datetime.datetime.fromtimestamp(m['time'])
            rec = {'id': m['id'], 'time': t.strftime('%Y-%m-%d %H:%M:%S'), 'message': m.get('message', '')}
            att = m.get('attachment')
            if att:
                try:
                    filename, body = read_attachment(att)
                    (d / filename).write_bytes(body); rec['attachment'] = filename
                    (d / '最新状态.json').write_bytes(body)
                    written.update([d / filename, d / '最新状态.json'])
                    dedicated = keep_direct_e002_receipt(d, body)
                    if dedicated is not None:
                        written.add(dedicated)
                except Exception as e:   # noqa: BLE001  附件过期（约 3 小时）就只留消息
                    rec['attachment_error'] = str(e)
            with open(d / '消息记录.jsonl', 'a', encoding='utf-8', newline='\n') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            written.add(d / '消息记录.jsonl')
            seen.add(m['id']); changed += 1
        seen_f.write_text(json.dumps(sorted(seen)), encoding='utf-8', newline='\n')
        written.add(seen_f)
        last = (d / '消息记录.jsonl').read_text('utf-8').splitlines()[-1] if (d / '消息记录.jsonl').exists() else None
        rows.append((name, json.loads(last) if last else None))
    derived_changed = False
    lines = ['# 实验队列最新状态' if all_sources else '# 甲乙最新状态', '', f'更新于 {datetime.datetime.now():%Y-%m-%d %H:%M}', '', '| 执行方 | 快照时间 | 状态与进度 |', '|---|---|---|']
    for name, r in rows:
        snapshot = REPO / name / '最新状态.json'
        when = r['time'] if r else '—'
        summary = r['message'] if r else '还没有消息'
        if snapshot.exists():
            try:
                status = status_for_folder(REPO / name)
                when = status['source_snapshot_at'] or when
                summary = status_summary(status)
                derived = REPO / name / '当前状态.json'
                body = json.dumps(status, ensure_ascii=False, indent=2) + '\n'
                if not derived.exists() or derived.read_text(encoding='utf-8') != body:
                    derived.write_text(body, encoding='utf-8', newline='\n')
                    written.add(derived)
                    derived_changed = True
            except (OSError, ValueError, TypeError) as exc:
                summary += f'；快照解析失败：{type(exc).__name__}'
        summary = summary.replace('|', '／').replace('\n', ' ')
        lines.append(f"| {DISPLAY_NAMES.get(name, name)} | {when} | {summary} |")
    scope_note = '外部同学只有甲、乙两位；原“同学丙”是组长本机3090代跑队列的内部标识，目录和消息话题沿用旧名。' if all_sources else '本页记录甲、乙的带时间快照；Codex已于11-10接管本机、服务器、甲乙的统一安排与监测。'
    lines += ['', scope_note, '', '状态以所列时间的快照为准；本轮失败按最近一次运行记录判定，保留的历史错误不代表当前故障。']
    for name, _ in rows:
        dedicated = REPO / name / '最近E002专用回执.json'
        if dedicated.exists():
            value = json.loads(dedicated.read_bytes())
            typed_status = runtime_status(value)
            count = typed_status.get('optimizer_updates')
            count = count if count is not None else '当前进程匹配步数未回报'
            lines += ['', f'{name}另有E002专用回执：{value.get("time")}，'
                      f'该次实际更新{count}。此证据独立保留；后续通用快照不覆盖它，'
                      '也不据旧回执认定当前仍在运行。']
    table_changed = not page.exists() or page.read_text(encoding='utf-8').splitlines()[4:] != lines[4:]
    if changed or table_changed:
        page.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
        written.add(page)
    print('\n'.join(lines))
    print(f'新消息 {changed} 条')
    if (changed or table_changed or derived_changed) and not a.no_push:
        paths = sorted(str(p.relative_to(REPO)) for p in written)
        subprocess.run(['git', '-C', str(REPO), 'add', '--', *paths], check=True)
        subprocess.run(['git', '-C', str(REPO), 'commit', '-q', '-m', f'状态存档 {datetime.datetime.now():%m-%d %H:%M}'], check=True)
        subprocess.run(['git', '-C', str(REPO), 'push', '-q'], check=True)
        print('已推送到 GitHub')


if __name__ == '__main__':
    main()
