#!/usr/bin/env python3
"""
Basic Roofline Analysis Example

Demonstrates how to use the roofline analyzer for LLM inference analysis.
"""

import sys
sys.path.insert(0, '..')

from src import (
    RooflineAnalyzer,
    T4Specs,
    MistralConfig,
    Llama3_8BConfig,
)


def main():
    print("=" * 60)
    print("Roofline Analysis Example")
    print("=" * 60)
    
    # Initialize analyzer with T4 GPU and Mistral-7B
    analyzer = RooflineAnalyzer(
        gpu_specs=T4Specs(),
        model_config=MistralConfig()
    )
    
    # Print summary table
    print("\n" + analyzer.summary_table())
    
    # Detailed analysis for specific workload
    print("\n" + "=" * 60)
    print("Detailed Analysis: 512 token prompt")
    print("=" * 60)
    
    prefill = analyzer.analyze(seq_len=512, phase="prefill")
    decode = analyzer.analyze(seq_len=512, phase="decode")
    
    print(f"\nPrefill Phase:")
    print(f"  FLOPs: {prefill.flops / 1e12:.2f} TFLOPS")
    print(f"  Memory: {prefill.memory_bytes / 1e9:.2f} GB")
    print(f"  Operational Intensity: {prefill.operational_intensity:.1f} FLOPs/Byte")
    print(f"  Bottleneck: {prefill.bottleneck}")
    print(f"  Theoretical Peak: {prefill.theoretical_tflops:.1f} TFLOPS")
    
    print(f"\nDecode Phase (per token):")
    print(f"  FLOPs: {decode.flops / 1e9:.2f} GFLOPS")
    print(f"  Memory: {decode.memory_bytes / 1e9:.2f} GB")
    print(f"  Operational Intensity: {decode.operational_intensity:.1f} FLOPs/Byte")
    print(f"  Bottleneck: {decode.bottleneck}")
    print(f"  Theoretical Peak: {decode.theoretical_tflops:.1f} TFLOPS")
    
    # Compare with Llama-3-8B
    print("\n" + "=" * 60)
    print("Comparison: Mistral-7B vs Llama-3-8B on T4")
    print("=" * 60)
    
    llama_analyzer = RooflineAnalyzer(
        gpu_specs=T4Specs(),
        model_config=Llama3_8BConfig()
    )
    
    for model_name, anlz in [("Mistral-7B", analyzer), ("Llama-3-8B", llama_analyzer)]:
        result = anlz.analyze(seq_len=512, phase="prefill")
        print(f"\n{model_name}:")
        print(f"  Prefill OI: {result.operational_intensity:.1f} FLOPs/Byte")
        print(f"  Bottleneck: {result.bottleneck}")
    
    # Generate roofline plot
    print("\n" + "=" * 60)
    print("Generating Roofline Plot...")
    print("=" * 60)
    
    fig = analyzer.plot_roofline(
        save_path="roofline_example.png"
    )
    print("✅ Saved to roofline_example.png")
    
    # Show optimization recommendations
    print("\n" + "=" * 60)
    print("Optimization Recommendations")
    print("=" * 60)
    
    print("""
    Based on the analysis:
    
    1. DECODE IS MEMORY-BOUND (OI ~1-2 FLOPs/Byte)
       - Use quantization (INT8/INT4) to reduce memory traffic
       - Increase batch size to amortize weight loading
       - Consider speculative decoding for higher throughput
    
    2. PREFILL CAN BE COMPUTE-BOUND (OI >200 FLOPs/Byte for long prompts)
       - Ensure Tensor Cores are utilized (FP16/BF16)
       - Use Flash Attention to reduce memory traffic
       - Consider chunked prefill for very long contexts
    
    3. HARDWARE SCALING
       - A100 (1.5 TB/s) would be ~4.7x faster for memory-bound decode
       - H100 (3.35 TB/s) would be ~10x faster for memory-bound decode
    """)


if __name__ == "__main__":
    main()
