<div align="center">

# Atlas AI
### Distributed AI Infrastructure · Transformer Systems · Performance Engineering

</div>

---

Atlas AI is an open machine learning systems platform focused on:

- Distributed training infrastructure
- Transformer systems engineering
- Inference optimization
- Performance benchmarking
- Observability and diagnostics
- AI infrastructure research

The project is built from first principles to explore how modern AI systems behave under real engineering constraints such as:

- Memory scaling
- Communication overhead
- Synchronization cost
- Inference latency
- Throughput efficiency
- Transformer cache growth

---

## Core Systems

### Deep Learning Infrastructure

- Reverse-mode autograd engine
- Neural network framework
- MLP training runtime
- Optimizer abstractions (SGD + Momentum)
- Checkpointing system

---

### Transformer Infrastructure

- Token embeddings
- Positional encoding
- Self-attention
- Transformer blocks
- Tiny transformer model
- KV-cache system
- Autoregressive decoding
- Streaming token generation

---

### Distributed Systems

- Multiprocessing distributed runtime
- Gradient synchronization
- Communication profiling
- Distributed scaling simulation
- Runtime scaling benchmarks

---

### Serving & Observability

- FastAPI inference server
- Inference latency monitoring
- Transformer memory diagnostics
- Dashboard metrics
- Historical benchmark persistence
- Regression detection infrastructure

---

### Performance Engineering

- Optimizer benchmarking
- Distributed runtime benchmarking
- Transformer generation benchmarking
- Memory profiling
- CI performance automation

---

## System Architecture

```mermaid
flowchart TB

    A[Atlas AI]

    A --> B[Training Infrastructure]
    A --> C[Transformer Stack]
    A --> D[Distributed Runtime]
    A --> E[Serving & Observability]
    A --> F[Performance Engineering]

    B --> B1[Autograd · Optimizers · Checkpointing]

    C --> C1[Embeddings · Attention · KV Cache]

    D --> D1[Workers · Gradient Sync · Communication]

    E --> E1[FastAPI · Metrics · Dashboard]

    F --> F1[Benchmarks · Regression Detection · CI]
```
---

## Architecture Overview

```text
                        ┌────────────────────┐
                        │  Transformer Stack │
                        │--------------------│
                        │ Embeddings         │
                        │ Positional Encoding│
                        │ Self-Attention     │
                        │ Transformer Blocks │
                        │ KV Cache           │
                        └─────────┬──────────┘
                                  │
                     ┌────────────▼────────────┐
                     │  Training Infrastructure │
                     │--------------------------│
                     │ Autograd Engine          │
                     │ Optimizers               │
                     │ Checkpointing            │
                     │ Experiment Tracking      │
                     └────────────┬─────────────┘
                                  │
                   ┌──────────────▼──────────────┐
                   │ Distributed Runtime          │
                   │------------------------------│
                   │ Multiprocessing Workers      │
                   │ Gradient Synchronization     │
                   │ Communication Profiling      │
                   │ Scaling Simulation           │
                   └──────────────┬──────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │ Serving & Observability   │
                    │---------------------------│
                    │ FastAPI Inference Server  │
                    │ Streaming Generation      │
                    │ Dashboard Metrics         │
                    │ Memory Diagnostics        │
                    │ Regression Detection      │
                    └───────────────────────────┘
```

---

## Benchmark Visualizations

### Optimizer Convergence

![Optimizer Benchmark](assets/benchmarks/optimizer_comparison.png)

---

### Distributed Runtime Scaling

![Distributed Scaling](assets/benchmarks/distributed_runtime_scaling.png)

---

### Transformer Generation Benchmark

![Transformer Benchmark](assets/benchmarks/transformer_generation_benchmark.png)

---

### Training Loss Convergence

![Loss Curve](assets/benchmarks/mlp_loss_curve.png)

---

## Infrastructure Screenshots

### FastAPI Inference API

![FastAPI Docs](assets/screenshots/fastapi-docs.png)

---

### Transformer Dashboard Metrics

![Dashboard Metrics](assets/screenshots/dashboard-api.png)

---

### Serving Metrics

![Serving Metrics](assets/screenshots/metrics-api.png)

---

### CI Performance Automation

![GitHub Actions CI](assets/screenshots/github-actions-ci.png)

---

### Transformer Benchmark Runtime

![Transformer Benchmark Runtime](assets/screenshots/transformer-benchmark-terminal.png)

---

### Streaming Token Generation

![Streaming Generation](assets/screenshots/streaming-generation.png)

---

## Example Capabilities

### Run Distributed Runtime

```bash
atlas run-distributed
```

### Train Transformer Components

```bash
atlas train
```

### Run Optimizer Benchmark

```bash
PYTHONPATH=. python scripts/benchmark_optimizers.py
```

### Run Transformer Generation Benchmark

```bash
PYTHONPATH=. python scripts/benchmark_transformer_generation.py
```

### Start Inference Server

```bash
uvicorn src.atlas_ai.serving.inference_server:app --host 0.0.0.0 --port 8000
```

---

## Example Serving Endpoints

### Inference API

```text
POST /predict
```

### Metrics API

```text
GET /metrics
```

### Dashboard API

```text
GET /dashboard
```

---

## Example Systems Diagnostics

### Distributed Runtime

```text
Atlas AI — Distributed Runtime
----------------------------------------
num_workers: 4
total_runtime_sec: 1.823

Worker Results:
worker_id=0 execution_time_sec=1.7293
worker_id=3 execution_time_sec=1.7462
worker_id=1 execution_time_sec=1.793
worker_id=2 execution_time_sec=1.8179
```

---

### Transformer Generation Benchmark

```text
tokens=1  latency=0.000113  throughput=8840.87
tokens=2  latency=0.000243  throughput=8235.43
tokens=4  latency=0.000505  throughput=7916.98
tokens=8  latency=0.001791  throughput=4465.70
tokens=16 latency=0.013770  throughput=1161.94
```

---

## Engineering Focus

Atlas AI focuses on:

- Distributed ML systems
- Transformer inference engineering
- Memory-aware infrastructure
- Performance benchmarking
- Observability systems
- Scalable AI serving
- Optimization diagnostics

---

## Current Research Areas

- KV-cache optimization
- Distributed synchronization overhead
- Transformer memory scaling
- Autoregressive inference latency
- Streaming generation systems
- Performance regression analysis

---

## Future Roadmap

### Distributed Training

- Asynchronous gradient synchronization
- Tensor parallelism
- Pipeline parallelism
- Optimizer state sharding

### Transformer Optimization

- FlashAttention-style kernels
- Quantization
- Efficient attention mechanisms
- Long-context optimization

### Infrastructure Engineering

- Prometheus integration
- Grafana dashboards
- Distributed tracing
- GPU profiling
- Cluster orchestration

---

## CI & Reliability

Atlas AI includes:

- Automated testing
- Benchmark CI workflows
- Regression detection
- Performance validation

GitHub Actions automatically:

- Runs tests
- Executes benchmarks
- Validates infrastructure behavior

---

## Technical Highlights

- 37+ automated tests
- Transformer inference benchmarking
- KV-cache simulation infrastructure
- Streaming token generation
- Distributed runtime execution
- Communication cost profiling
- Historical benchmark persistence
- Performance regression detection
- CI benchmark automation
- Transformer memory diagnostics

---

## Project Philosophy

Atlas AI treats machine learning as a systems engineering problem.

The project focuses on understanding how:

- Communication affects scaling
- Memory constrains transformers
- Inference latency impacts serving
- Optimization changes convergence
- Observability improves reliability

rather than only maximizing model accuracy.

---

## License

MIT

---

<div align="center">

*Omprakash Sahani — ML Systems Engineer (Distributed Training · Optimization · Systems)*

</div>
