"""Regression cases from reported runner completion and recovery logs."""
import unittest
import datetime
import json
from pathlib import Path
import tempfile

from collect import runtime_status, keep_direct_e002_receipt, status_for_folder, process_created


class RuntimeStatusTests(unittest.TestCase):
    def status(self, *lines):
        return runtime_status({'runner_log_tail': list(lines), 'failures': {'old': 'retained historical marker'}})

    def test_successful_queue_says_no_errors(self):
        result = self.status('10-26 21:41:15 结束 trial，返回码 0', '10-26 21:41:15 全部任务结束，没有出错')
        self.assertEqual(result['state'], 'completed')
        self.assertEqual(result['current_run_failed_tasks'], [])

    def test_restart_supersedes_old_failure(self):
        result = self.status('10-26 20:27:41 结束 trial，返回码 1',
                             '10-26 20:34:48 任务文件 同学甲.json',
                             '10-26 20:39:25 结束 trial，返回码 0',
                             '10-26 20:51:18 开始 formal')
        self.assertEqual(result['state'], 'running')
        self.assertEqual(result['current_run_failed_tasks'], [])
        self.assertEqual(result['historical_failure_events'], 1)

    def test_failed_queue_retains_failed_task(self):
        result = self.status('10-26 20:27:41 结束 trial，返回码 1', "10-26 20:27:46 全部任务结束；出错的：['trial']")
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(result['current_run_failed_tasks'], ['trial'])

    def test_empty_failure_summary_is_success(self):
        self.assertEqual(self.status('10-26 20:27:46 全部任务结束；出错的：[]')['state'], 'completed')

    def test_one_success_is_not_whole_queue_completion(self):
        self.assertEqual(self.status('10-26 20:27:46 结束 trial，返回码 0')['state'], 'between_tasks')

    def test_successful_retry_clears_current_failure(self):
        result = self.status('10-26 20:27:41 结束 trial，返回码 1',
                             '10-26 20:34:48 开始 trial', '10-26 20:39:25 结束 trial，返回码 0',
                             '10-26 20:39:25 全部任务结束，没有出错')
        self.assertEqual(result['state'], 'completed')
        self.assertEqual(result['current_run_failed_tasks'], [])

    def test_other_task_dry_run_does_not_complete_or_fail_reported_queue(self):
        for ending in ('全部任务结束，没有出错', "全部任务结束；出错的：['foreign']"):
            with self.subTest(ending=ending):
                result = runtime_status({
                    'task_file': '同学丙.json',
                    'runner_log_tail': [
                        '10-26 17:56:01 任务文件 同学丙.json（同学丙），混合精度 bf16',
                        '10-26 22:31:17 开始 unet_r34_12b_s20261017',
                        '10-26 23:15:26 任务文件 本机丙_第三批.json（本机3090），混合精度 fp16',
                        '10-26 23:15:26 ' + ending]})
                self.assertEqual(result['state'], 'unknown')
                self.assertTrue(result['log_context_mismatch'])
                self.assertEqual(result['current_run_failed_tasks'], [])


class E002DedicatedReceiptTests(unittest.TestCase):
    def test_powershell_roundtrip_datetime_100ns_precision(self):
        created = process_created({'CreationDate': '2026-11-12T08:54:32.6390540+08:00'})
        self.assertAlmostEqual(created, 1794444872.639054, places=6)

    def fixture(self):
        # Shape and counters from the actual 14:27 micro1 report; no dependency
        # on the locally collected, untracked peer attachment.
        value = {
            '指令编号': {'编号': 'yi-e002-recover-v2-1112-1340'},
            '当前活动目录': r'C:\runs\foldIII_micro1',
            '训练进程': [{'ProcessId': 2072, 'CreationDate': '/Date(1794462902460)/',
                         'CommandLine': r'python e002_train.py --micro 1 --out C:\runs\foldIII_micro1'}],
            '最近事件': [{'PID': 2072, 'stage': 'training', 'optimizer_updates': n}
                         for n in [1090, 1100, 1110, 1120]],
        }
        anchor = datetime.datetime.now().astimezone()
        value['time'] = anchor.isoformat()
        for offset, event in zip([58, 39, 20, 1], value['最近事件']):
            event['time'] = (anchor - datetime.timedelta(seconds=offset)).isoformat()
        return value

    def test_object_identity_and_new_micro1_events_are_recognized(self):
        result = runtime_status(self.fixture())
        self.assertEqual(result['state'], 'running')
        self.assertEqual(result['optimizer_updates'], 1120)
        self.assertEqual(result['trainer_PID'], 2072)
        self.assertEqual(result['trainer_created_unix'], 1794462902.46)
        self.assertTrue(result['command_matches_reported_run'])

    def test_old_pid_events_do_not_supply_new_trainer_progress(self):
        value = self.fixture()
        for event in value['最近事件']:
            event['PID'] = 19660
        result = runtime_status(value)
        self.assertEqual(result['state'], 'unknown')
        self.assertIsNone(result['optimizer_updates'])

    def test_reused_pid_event_predating_creation_is_rejected(self):
        value = self.fixture()
        created_ms = int((datetime.datetime.now().timestamp() + 1) * 1000)
        value['训练进程'][0]['CreationDate'] = f'/Date({created_ms})/'
        self.assertEqual(runtime_status(value)['state'], 'unknown')

    def test_wrong_command_or_run_is_rejected(self):
        value = self.fixture()
        value['当前活动目录'] = 'wrong_other_run'
        self.assertEqual(runtime_status(value)['state'], 'unknown')

    def test_legacy_snapshot_does_not_erase_dedicated_but_new_idle_conflicts(self):
        value = self.fixture()
        anchor = datetime.datetime.fromisoformat(value['time'])
        legacy = {'time': (anchor + datetime.timedelta(seconds=1)).isoformat(),
                  '指令编号': 'yi-st3d-1110-v2',
                  '机器': {'GPU': 'NVIDIA TITAN RTX, 6681 MiB, 24576 MiB, 96 %, 88'}}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.assertIsNotNone(keep_direct_e002_receipt(folder, json.dumps(value).encode()))
            self.assertIsNone(keep_direct_e002_receipt(folder, json.dumps(legacy).encode()))
            (folder / '最新状态.json').write_text(json.dumps(legacy), encoding='utf-8')
            current = status_for_folder(folder)
            self.assertEqual(current['optimizer_updates'], 1120)
            self.assertEqual(current['source_snapshot_at'], value['time'])
            legacy['机器']['GPU'] = 'NVIDIA TITAN RTX, 0 MiB, 24576 MiB, 0 %, 36'
            (folder / '最新状态.json').write_text(json.dumps(legacy), encoding='utf-8')
            current = status_for_folder(folder)
            self.assertNotEqual(current['state'], 'running')
            self.assertTrue(current['conflicting_newer_idle_hardware'])

    def test_future_nested_task_mention_is_not_direct_e002_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            value = self.fixture()
            identity = value.pop('指令编号')
            value['指令编号'] = 'yi-st3d-1110-v2'
            value['future_task'] = identity
            self.assertIsNone(keep_direct_e002_receipt(Path(directory), json.dumps(value).encode()))


if __name__ == '__main__':
    unittest.main()
