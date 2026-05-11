import statistics
import time

import requests


URL = "http://127.0.0.1:8000/predict"


def main():

    latencies = []

    payload = [1.0, 2.0, 3.0]

    num_requests = 20

    for _ in range(num_requests):

        start = time.perf_counter()

        response = requests.post(
            URL,
            json=payload,
        )

        end = time.perf_counter()

        response.raise_for_status()

        latencies.append(end - start)

    average_latency = statistics.mean(
        latencies
    )

    throughput = (
        num_requests / sum(latencies)
    )

    print("Atlas AI — Inference Benchmark")
    print("-" * 40)

    print(
        f"Requests: {num_requests}"
    )

    print(
        f"Average Latency: "
        f"{average_latency:.6f} sec"
    )

    print(
        f"Min Latency: "
        f"{min(latencies):.6f} sec"
    )

    print(
        f"Max Latency: "
        f"{max(latencies):.6f} sec"
    )

    print(
        f"Throughput: "
        f"{throughput:.2f} req/sec"
    )


if __name__ == "__main__":
    main()
