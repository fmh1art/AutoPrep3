import multiprocessing
from collections import deque
from threading import Lock
import time


class MultiProcesser:
    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.pool = multiprocessing.Pool(processes=num_processes)
        self.results = []
        self._closed = False

    def submit_task(self, func, *args, **kwargs):
        result = self.pool.apply_async(func, args=args, kwds=kwargs)
        self.results.append(result)

    def wait_for_completion(self, timeout=None):
        if self._closed:
            return []

        self.pool.close()
        results = []
        has_timeout_or_error = False
        for i, result in enumerate(self.results):
            try:
                if timeout:
                    results.append(result.get(timeout=timeout))
                else:
                    results.append(result.get())
            except Exception as e:
                print(f"Error getting result {i}: {e}")
                results.append(None)  # 或者抛出异常
                has_timeout_or_error = True

        if has_timeout_or_error:
            # Avoid hanging on join() when one or more workers are stuck.
            self.pool.terminate()

        self.pool.join()
        self._closed = True
        return results

    def close(self):
        if self.pool and not self._closed:
            self.pool.terminate()
            self.pool.join()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class DynamicMultiProcesser:
    """
    动态进程池管理器
    始终保持有 num_processes 个活跃进程在运行
    当一个进程完成时，自动启动新的进程
    """
    def __init__(self, num_processes, verbose=False):
        self.num_processes = num_processes
        # 设置进程池大小为 num_processes，但我们只会同时运行最多 num_processes 个任务
        self.pool = multiprocessing.Pool(processes=num_processes)
        self.pending_tasks = deque()  # 待处理的任务队列
        self.active_results = []  # 当前活跃的任务结果对象
        self.completed_results = []  # 已完成的任务结果
        self.global_lock = Lock()  # 全局锁保护所有共享状态
        self.verbose = verbose
        self._closed = False

    def submit_task(self, func, *args, **kwargs):
        """提交任务到待处理队列"""
        with self.global_lock:
            self.pending_tasks.append((func, args, kwargs))
            if self.verbose:
                print(f"Task submitted. Pending: {len(self.pending_tasks)}, Active: {len(self.active_results)}")

    def _get_next_task(self):
        """线程安全地获取下一个待处理任务"""
        with self.global_lock:
            if self.pending_tasks:
                task = self.pending_tasks.popleft()
                if self.verbose:
                    print(f"Retrieved task. Pending: {len(self.pending_tasks)}, Active: {len(self.active_results)}")
                return task
            return None

    def _submit_next_task(self):
        """提交下一个待处理任务到进程池"""
        task = self._get_next_task()
        if task:
            func, args, kwargs = task
            result = self.pool.apply_async(func, args=args, kwds=kwargs)
            with self.global_lock:
                self.active_results.append(result)
                if self.verbose:
                    print(f"Task submitted to pool. Active: {len(self.active_results)}")
            return True
        return False

    def _check_completed_tasks(self):
        """检查并处理已完成的任务"""
        completed_indices = []
        completed_count = 0

        with self.global_lock:
            # 复制活跃结果列表以避免在迭代时修改
            active_copy = list(enumerate(self.active_results))

        for i, result in active_copy:
            if result.ready():
                try:
                    result_value = result.get(timeout=1.0)  # 添加超时以避免阻塞
                    with self.global_lock:
                        self.completed_results.append(result_value)
                        completed_indices.append(i)
                        completed_count += 1
                    if self.verbose:
                        print(f"Task completed successfully. Total completed: {len(self.completed_results)}")
                except Exception as e:
                    if self.verbose:
                        print(f"Task failed with error: {e}")
                    with self.global_lock:
                        completed_indices.append(i)

        # 移除已完成的任务（从后往前移除以保持索引正确）
        with self.global_lock:
            for i in reversed(completed_indices):
                if i < len(self.active_results):
                    del self.active_results[i]

        return completed_count

    def _fill_active_slots(self):
        """填充活跃任务槽位"""
        submitted_count = 0
        with self.global_lock:
            active_count = len(self.active_results)
            pending_count = len(self.pending_tasks)

        while active_count + submitted_count < self.num_processes and pending_count - submitted_count > 0:
            if self._submit_next_task():
                submitted_count += 1
            else:
                break

        if self.verbose and submitted_count > 0:
            print(f"Filled {submitted_count} active slots")

        return submitted_count

    def process_all_tasks(self):
        """
        处理所有任务，动态维护活跃进程数量
        返回所有完成的结果
        """
        if self.verbose:
            print(f"Starting dynamic processing with {self.num_processes} processes")

        # 首先启动初始的任务（不超过进程池容量）
        initial_count = 0
        with self.global_lock:
            available_slots = min(self.num_processes, len(self.pending_tasks))

        for _ in range(available_slots):
            if self._submit_next_task():
                initial_count += 1

        if self.verbose:
            print(f"Initial {initial_count} tasks submitted")

        # 持续监控和补充进程
        while True:
            with self.global_lock:
                has_active = len(self.active_results) > 0
                has_pending = len(self.pending_tasks) > 0

            if not has_active and not has_pending:
                break

            # 检查已完成的任务
            completed = self._check_completed_tasks()

            # 补充新的任务到活跃队列
            submitted = self._fill_active_slots()

            # 如果还有活跃任务且没有新任务提交，等待更长时间
            if has_active and submitted == 0 and completed == 0:
                if self.verbose:
                    print("Waiting for tasks to complete...")
                time.sleep(0.5)  # 增加等待时间
            elif has_active:
                time.sleep(0.2)  # 有活动时稍微等待
            else:
                time.sleep(0.1)  # 没有活动时快速检查

        if self.verbose:
            print(f"All tasks completed. Total results: {len(self.completed_results)}")

        # 关闭进程池
        self._shutdown(normal=True)

        return self.completed_results

    def _shutdown(self, normal=False):
        if self.pool and not self._closed:
            if normal:
                self.pool.close()
            else:
                self.pool.terminate()
            self.pool.join()
            self._closed = True

    def close(self):
        self._shutdown(normal=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def test_func(a, b):
    sum_result = a + b
    product_result = a * b
    return sum_result, product_result


def slow_task(task_id, duration):
    """模拟一个耗时的任务 - 必须在模块级别定义以便pickle"""
    import time
    print(f"Task {task_id} started, will run for {duration}s")
    time.sleep(duration)
    result = f"Task {task_id} completed after {duration}s"
    print(result)
    return result


def test_dynamic_processor():
    """测试动态进程管理器"""
    print("Testing DynamicMultiProcesser...")

    # 创建动态处理器
    processor = DynamicMultiProcesser(num_processes=3, verbose=True)

    # 提交一些任务
    tasks = [
        (slow_task, 1, 2),
        (slow_task, 2, 1),
        (slow_task, 3, 3),
        (slow_task, 4, 1),
        (slow_task, 5, 2),
        (slow_task, 6, 1),
    ]

    for task in tasks:
        processor.submit_task(*task)

    # 处理所有任务
    results = processor.process_all_tasks()

    print(f"\nAll tasks completed! Results: {len(results)}")
    for result in results:
        print(f"Result: {result}")

