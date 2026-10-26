"""Regression cases from reported runner completion and recovery logs."""
import unittest

from collect import runtime_status


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


if __name__ == '__main__':
    unittest.main()
