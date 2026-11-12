"""Read-only E002 provenance and bounded live telemetry; no weights or dataset.

Run with the existing Python environment. Writes one small JSON for the normal
peer status channel. It never starts/stops training or changes system settings.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import time


def now():
    return datetime.datetime.now().astimezone().isoformat()


def run(argv, env=None, timeout=15):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, env=env)
        return {"argv": argv, "returncode": p.returncode,
                "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": argv, "error": type(error).__name__ + ": " + str(error)}


def small_file(path):
    if not path.is_file():
        return {"path": str(path), "exists": False}
    before = path.stat()
    if before.st_size > 256 * 1024:
        return {"path": str(path), "bytes": before.st_size,
                "text_included": False, "reason": "Use a small receipt, not a weight or full training log"}
    data = path.read_bytes()
    after = path.stat()
    return {"path": str(path), "bytes": len(data),
            "SHA256": hashlib.sha256(data).hexdigest(),
            "stable_during_read": (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "text": data.decode("utf-8-sig", errors="replace")}


def windows_context(out, since):
    if os.name != "nt":
        return {"platform": os.name, "Windows_events": "not applicable"}
    env = os.environ.copy()
    env["E002_EVIDENCE_OUT"] = str(out)
    env["E002_EVIDENCE_SINCE"] = since
    # The target path is passed as a process environment value, not interpolated
    # into PowerShell source. These are OS queries, not UI automation.
    code = r'''
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$target=$env:E002_EVIDENCE_OUT.ToLowerInvariant()
$result=[ordered]@{}
try {
 $result.processes=@(CimCmdlets\Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
 Where-Object { $_.CommandLine -and $_.CommandLine.ToLowerInvariant().Contains($target) } |
 ForEach-Object { [ordered]@{PID=$_.ProcessId;ParentPID=$_.ParentProcessId;CreationTime=$_.CreationDate.ToString('o');CommandLine=$_.CommandLine} })
} catch {$result.process_query_error=$_.Exception.Message}
try {
 $result.tasks=@(Get-ScheduledTask | Where-Object {($_.Actions | Where-Object {$_.Arguments -match 'e002_supervise|e002_train'})} |
 ForEach-Object {$task=$_;$info=$task|Get-ScheduledTaskInfo;[ordered]@{Name=$task.TaskName;State=[string]$task.State;Actions=$task.Actions;ExecutionTimeLimit=[string]$task.Settings.ExecutionTimeLimit;MultipleInstances=[string]$task.Settings.MultipleInstances;LastTaskResult=$info.LastTaskResult;LastRunTime=$info.LastRunTime.ToString('o')}})
} catch {$result.task_query_error=$_.Exception.Message}
try {
 $result.GPU_system_events=@(Get-WinEvent -FilterHashtable @{LogName='System';StartTime=[datetime]$env:E002_EVIDENCE_SINCE;ProviderName=@('nvlddmkm','Display','Microsoft-Windows-WHEA-Logger')} -MaxEvents 200 -ErrorAction Stop |
 ForEach-Object {[ordered]@{Time=$_.TimeCreated.ToString('o');Provider=$_.ProviderName;ID=$_.Id;RecordID=$_.RecordId;Message=$_.Message;XML=$_.ToXml()}})
} catch {$result.event_query_status=$_.Exception.Message}
try {$mem=CimCmdlets\Get-CimInstance Win32_OperatingSystem;$result.memory=@{total_bytes=[long]$mem.TotalVisibleMemorySize*1024;available_bytes=[long]$mem.FreePhysicalMemory*1024}} catch {$result.memory_query_error=$_.Exception.Message}
$result|ConvertTo-Json -Depth 8 -Compress
'''
    result = run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", code], env=env, timeout=25)
    if result.get("returncode") == 0:
        try:
            result["parsed"] = json.loads(result.pop("stdout"))
        except json.JSONDecodeError:
            pass
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--kit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--since", default="2026-11-12T11:15:00+08:00")
    p.add_argument("--instruction-id", default="")
    a = p.parse_args()
    if a.duration < 0 or a.duration > 300:
        p.error("duration must be between0 and300 seconds")
    out, kit = a.out.resolve(), a.kit.resolve()
    config_path = out / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    files = {name: small_file(out / name) for name in (
        "config.json", "continuation_parent_receipt.json", "validation.jsonl", "first100_updates.json")}
    matches = {}
    for name, expected in config["source_hashes"].items():
        if Path(name).name != name:
            raise ValueError("source hash keys must be filenames")
        source = kit / name
        actual = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
        matches[name] = {"expected": expected, "actual": actual, "matches": expected == actual}
    disk = shutil.disk_usage(out)
    samples = []
    start = time.monotonic()
    while True:
        observed = now()
        progress = json.loads((out / "progress.json").read_text(encoding="utf-8-sig"))
        gpu = run(["nvidia-smi", "--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu,fan.speed,clocks.sm,power.draw,power.limit,clocks_throttle_reasons.active",
                   "--format=csv,noheader,nounits"], timeout=8)
        samples.append({"time": observed, "progress": progress, "GPU": gpu})
        remaining = a.duration - (time.monotonic() - start)
        if remaining <= 0:
            break
        time.sleep(min(5, remaining))
    context = windows_context(out, a.since)
    processes = (context.get("parsed") or {}).get("processes") or []
    trainers = [{"ProcessId": q["PID"], "ParentProcessId": q["ParentPID"],
                 "CreationDate": q["CreationTime"], "CommandLine": q["CommandLine"]}
                for q in processes if "e002_train.py" in q.get("CommandLine", "").lower()]
    receipt = {
        "time": now(), "out": str(out), "kit": str(kit),
        "instruction_id": a.instruction_id,
        "当前活动目录": str(out), "训练进程": trainers,
        "training_progress": samples[-1]["progress"],
        "files": files, "frozen_source_checks": matches,
        "all_current_sources_match_runconfig": all(q["matches"] for q in matches.values()),
        "actual_micro": config["micro_batch"], "actual_accumulation": config["accumulation"],
        "effective_batch": config["global_batch"],
        "telemetry_elapsed_seconds": time.monotonic() - start, "samples": samples,
        "disk": {"total_bytes": disk.total, "free_bytes": disk.free},
        "OS_context": context,
        "model_or_dataset_loaded": False, "training_or_settings_modified": False,
        "script_SHA256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "decision": "Keep healthy training. Validate receipt/source/creation and new events at the receiver; missing fields remain missing rather than being inferred.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8", newline="\n") as dest:
        json.dump(receipt, dest, ensure_ascii=False, indent=2)
    print(json.dumps({"receipt": str(a.output), "bytes": a.output.stat().st_size,
                      "sources_match": receipt["all_current_sources_match_runconfig"],
                      "samples": len(samples), "modified_training": False}))


if __name__ == "__main__":
    main()
