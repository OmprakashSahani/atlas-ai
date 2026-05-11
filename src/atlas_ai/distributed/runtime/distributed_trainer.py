import multiprocessing as mp
import time


def worker_process(
    worker_id: int,
    results_queue,
):
    """
    Simulate distributed worker computation.
    """

    start_time = time.perf_counter()

    total = 0

    for i in range(10_000_000):
        total += i

    end_time = time.perf_counter()

    execution_time = (
        end_time - start_time
    )

    results_queue.put(
        {
            "worker_id": worker_id,
            "execution_time_sec": round(
                execution_time,
                4,
            ),
            "result": total,
        }
    )


class DistributedTrainer:
    """
    Simple multiprocessing-based
    distributed runtime.
    """

    def __init__(
        self,
        num_workers: int = 4,
    ):
        self.num_workers = num_workers

    def run(self):

        results_queue = mp.Queue()

        processes = []

        start_time = time.perf_counter()

        for worker_id in range(
            self.num_workers
        ):

            process = mp.Process(
                target=worker_process,
                args=(
                    worker_id,
                    results_queue,
                ),
            )

            process.start()

            processes.append(process)

        results = []

        for _ in range(self.num_workers):

            results.append(
                results_queue.get()
            )

        for process in processes:
            process.join()

        end_time = time.perf_counter()

        total_runtime = (
            end_time - start_time
        )

        return {
            "num_workers": (
                self.num_workers
            ),
            "total_runtime_sec": round(
                total_runtime,
                4,
            ),
            "worker_results": results,
        }
