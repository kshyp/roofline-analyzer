# Roofline Analyzer

A toolkit for roofline performance analysis of LLM inference workloads on GPU.

## What is Roofline Analysis?

Roofline analysis helps you understand whether your workload is **compute-bound** or **memory-bound** by plotting achieved performance against operational intensity (FLOPs/Byte).

![Roofline Example](docs/roofline_example.png)

- **Memory-bound** (left of ridge point): Performance limited by memory bandwidth
- **Compute-bound** (right of ridge point): Performance limited by peak FLOPS

## Key Findings for LLM Inference

| Phase | Operational Intensity | Bottleneck | Optimization |
|-------|----------------------|------------|--------------|
| Prefill (short) | ~50-100 F/B | Memory | Batching, quantization |
| Prefill (long) | ~200-2000 F/B | Compute | Flash Attention, Tensor Cores |
| Decode | ~1-2 F/B | Memory | Quantization, speculative decoding |

## Quick Start

### Google Colab (Recommended)

1. Open `notebooks/roofline_analysis_t4.ipynb` in Colab
2. Select T4 GPU runtime: `Runtime → Change runtime type → T4 GPU`
3. Run all cells

### Local Installation

```bash
git clone https://github.com/YOUR_USERNAME/roofline-analyzer.git
cd roofline-analyzer
pip install -r requirements.txt
```

### Basic Usage

```python
from src.roofline import RooflineAnalyzer, T4Specs, MistralConfig

# Initialize analyzer
analyzer = RooflineAnalyzer(
    gpu_specs=T4Specs(),
    model_config=MistralConfig()
)

# Calculate operational intensity for prefill
oi = analyzer.get_operational_intensity(seq_len=512, phase="prefill")
print(f"Prefill OI: {oi:.1f} FLOPs/Byte")

# Plot roofline with your measurements
analyzer.plot_roofline(
    measured_points=[
        {"name": "My Model", "oi": 150, "tflops": 12}
    ],
    save_path="my_roofline.png"
)
```

## Project Structure

```
roofline-analyzer/
├── src/
│   ├── __init__.py
│   ├── roofline.py      # Core roofline analysis
│   ├── gpu_specs.py     # GPU hardware specifications
│   ├── model_configs.py # LLM architecture configs
│   └── benchmark.py     # Inference benchmarking
├── notebooks/
│   └── roofline_analysis_t4.ipynb  # Interactive Colab notebook
├── examples/
│   └── basic_analysis.py
├── requirements.txt
└── README.md
```

## Supported GPUs

| GPU | Peak FP16 (TFLOPS) | Memory BW (GB/s) | Ridge Point (F/B) |
|-----|-------------------|------------------|-------------------|
| T4 | 65 | 320 | 203 |
| V100 | 125 | 900 | 139 |
| A100 40GB | 312 | 1555 | 201 |
| A100 80GB | 312 | 2039 | 153 |
| H100 | 990 | 3350 | 296 |
| L4 | 121 | 300 | 403 |

## Supported Models

- Mistral-7B
- Llama-2-7B / 13B / 70B
- Llama-3-8B / 70B
- Custom (define your own config)

## Understanding the Results

### Memory-Bound Workloads (Decode)

When decode is memory-bound (OI < ridge point):
- **Throughput scales with memory bandwidth**, not compute
- A100 (1.5 TB/s) is ~4.7x faster than T4 (320 GB/s)
- Quantization helps by reducing bytes transferred

### Compute-Bound Workloads (Long Prefill)

When prefill is compute-bound (OI > ridge point):
- **Throughput scales with FLOPS**
- Tensor Cores utilization matters
- Flash Attention reduces memory traffic, keeping you compute-bound

## Batch Size Scaling

Batching improves throughput for memory-bound workloads:

```
Batch 1: Weights loaded once, used for 1 request
Batch 4: Weights loaded once, used for 4 requests → 4x better weight reuse
```

## Integration with inference-optimizer

This repo complements [inference-optimizer](https://github.com/kshyp/inference-optimizer) by adding roofline analysis to understand *why* you see the performance metrics you measure.

```python
# Combined workflow
from inference_optimizer.monitor import GpuMonitor
from roofline_analyzer import RooflineAnalyzer

with GpuMonitor() as monitor:
    result = runner.generate(prompt, max_new_tokens=100)

# Understand the bottleneck
analyzer = RooflineAnalyzer(gpu_specs=T4Specs(), model_config=MistralConfig())
analysis = analyzer.analyze(
    prompt_len=len(tokens),
    gen_tokens=100,
    measured_time=result.time
)
print(analysis.bottleneck)  # "memory-bound" or "compute-bound"
```

## References

- [Roofline Model (Williams et al.)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
- [LLM Inference Performance Engineering](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)

## License

MIT
