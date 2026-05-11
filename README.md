# Atlas AI

**Atlas AI** is an open ML systems platform for training, profiling, evaluating, and serving AI models.

The goal of this project is to study how machine learning systems behave under real engineering constraints such as compute, memory, communication, latency, and reproducibility.

## Current Status

Atlas AI currently includes:

- Python package structure
- Command-line interface
- Basic system information collection
- Automated test setup

## Quick Start

```bash
python -m pip install -e ".[dev]"
atlas
pytest
````

## Current CLI Output

```text
Atlas AI — System Info
----------------------------
python_version: 3.12.1
platform: Linux-6.8.0-1044-azure-x86_64-with-glibc2.39
processor: x86_64
machine: x86_64
```

## Planned Modules

| Module                | Purpose                                                   |
| --------------------- | --------------------------------------------------------- |
| System Profiler       | Inspect CPU, memory, GPU, Python, and runtime environment |
| Experiment Tracker    | Log runs, metrics, configs, and artifacts                 |
| Training Engine       | Train small models from first principles                  |
| Distributed Simulator | Analyze communication and synchronization bottlenecks     |
| Inference Runtime     | Serve models and measure latency/throughput               |
| Evaluation Engine     | Compare model quality and system performance              |
| Dashboard             | Visualize experiments, bottlenecks, and benchmarks        |

## Engineering Focus

Atlas AI focuses on:

* reproducible ML experiments
* measurable system performance
* distributed training behavior
* memory and communication bottlenecks
* practical AI infrastructure design

## License

MIT
