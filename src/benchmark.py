"""
Inference Benchmarking Module

Benchmarks actual LLM inference performance for comparison
with theoretical roofline analysis.
"""

import gc
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    prompt_len: int
    gen_tokens: int
    batch_size: int
    avg_time: float
    std_time: float
    tokens_per_sec: float
    achieved_tflops: float
    prefill_flops: float
    decode_flops: float
    total_flops: float


class InferenceBenchmark:
    """
    Benchmark LLM inference performance.
    
    Example:
        benchmark = InferenceBenchmark(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            load_in_4bit=True
        )
        benchmark.load()
        
        results = benchmark.run(
            prompt_lengths=[64, 128, 256],
            gen_tokens=32
        )
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        device_map: str = "auto",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for benchmarking")
        
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.device_map = device_map
        
        self.model = None
        self.tokenizer = None
        self.model_config = None
    
    def load(self) -> None:
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading tokenizer for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading model...")
        quantization_msg = ""
        if self.load_in_4bit:
            quantization_msg = " (4-bit)"
        elif self.load_in_8bit:
            quantization_msg = " (8-bit)"
        print(f"  Quantization:{quantization_msg or ' FP16'}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            device_map=self.device_map,
        )
        self.model.eval()
        
        # Extract model config for FLOP calculations
        self._extract_model_config()
        
        mem_used = torch.cuda.memory_allocated() / 1e9
        print(f"✅ Model loaded! GPU memory: {mem_used:.2f} GB")
    
    def _extract_model_config(self) -> None:
        """Extract model configuration for FLOP calculations."""
        from .model_configs import ModelConfig
        
        config = self.model.config
        
        self.model_config = ModelConfig(
            name=self.model_name.split("/")[-1],
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads),
            head_dim=config.hidden_size // config.num_attention_heads,
            vocab_size=config.vocab_size,
            max_position_embeddings=getattr(config, 'max_position_embeddings', 4096),
        )
    
    def _calculate_flops(self, prompt_len: int, gen_tokens: int, batch_size: int = 1) -> Dict[str, float]:
        """Calculate theoretical FLOPs for the benchmark."""
        from .roofline import RooflineAnalyzer
        
        analyzer = RooflineAnalyzer(model_config=self.model_config)
        
        prefill_flops = analyzer.calculate_prefill_flops(prompt_len, batch_size).total
        
        # Decode FLOPs: sum over all generated tokens
        decode_flops = 0
        for i in range(gen_tokens):
            kv_len = prompt_len + i
            decode_flops += analyzer.calculate_decode_flops(kv_len, batch_size).total
        
        return {
            "prefill": prefill_flops,
            "decode": decode_flops,
            "total": prefill_flops + decode_flops,
        }
    
    def run(
        self,
        prompt_lengths: List[int] = [64, 128, 256, 512],
        gen_tokens: int = 32,
        batch_size: int = 1,
        warmup_runs: int = 2,
        benchmark_runs: int = 5,
    ) -> List[BenchmarkResult]:
        """
        Run inference benchmarks.
        
        Args:
            prompt_lengths: List of prompt lengths to test
            gen_tokens: Number of tokens to generate per request
            batch_size: Batch size for inference
            warmup_runs: Number of warmup iterations
            benchmark_runs: Number of timed iterations
        
        Returns:
            List of BenchmarkResult objects
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        results = []
        
        for prompt_len in prompt_lengths:
            print(f"\nBenchmarking prompt_len={prompt_len}, batch_size={batch_size}...")
            
            # Create prompt
            prompt = self._create_prompt(prompt_len)
            inputs = self.tokenizer(
                [prompt] * batch_size,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=prompt_len
            )
            inputs = {k: v.cuda() for k, v in inputs.items()}
            actual_prompt_len = inputs['input_ids'].shape[1]
            
            # Warmup
            for _ in range(warmup_runs):
                with torch.no_grad():
                    _ = self.model.generate(
                        **inputs,
                        max_new_tokens=gen_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
            
            torch.cuda.synchronize()
            
            # Benchmark runs
            times = []
            for _ in range(benchmark_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=gen_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            # Calculate metrics
            total_tokens = batch_size * gen_tokens
            tokens_per_sec = total_tokens / avg_time
            
            flops = self._calculate_flops(actual_prompt_len, gen_tokens, batch_size)
            achieved_tflops = (flops['total'] / avg_time) / 1e12
            
            result = BenchmarkResult(
                prompt_len=actual_prompt_len,
                gen_tokens=gen_tokens,
                batch_size=batch_size,
                avg_time=avg_time,
                std_time=std_time,
                tokens_per_sec=tokens_per_sec,
                achieved_tflops=achieved_tflops,
                prefill_flops=flops['prefill'],
                decode_flops=flops['decode'],
                total_flops=flops['total'],
            )
            results.append(result)
            
            print(f"  Actual prompt: {actual_prompt_len} tokens")
            print(f"  Time: {avg_time:.3f}s ± {std_time:.3f}s")
            print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")
            print(f"  Achieved: {achieved_tflops:.2f} TFLOPS")
            
            # Cleanup
            gc.collect()
            torch.cuda.empty_cache()
        
        return results
    
    def _create_prompt(self, target_length: int) -> str:
        """Create a prompt of approximately the target token length."""
        base = "Please explain the following concept in detail: "
        padding = "example " * (target_length // 2)
        return base + padding
    
    def run_batch_scaling(
        self,
        batch_sizes: List[int] = [1, 2, 4],
        prompt_len: int = 128,
        gen_tokens: int = 16,
    ) -> List[Dict]:
        """
        Test how throughput scales with batch size.
        
        Returns:
            List of dicts with batch_size, time, throughput, memory
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        print("\n" + "=" * 60)
        print("BATCH SIZE SCALING ANALYSIS")
        print("=" * 60)
        
        results = []
        
        for batch_size in batch_sizes:
            try:
                # Clear memory
                gc.collect()
                torch.cuda.empty_cache()
                
                mem_before = torch.cuda.memory_allocated() / 1e9
                
                # Create batched input
                prompt = self._create_prompt(prompt_len)
                inputs = self.tokenizer(
                    [prompt] * batch_size,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=prompt_len
                )
                inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Warmup
                with torch.no_grad():
                    _ = self.model.generate(
                        **inputs,
                        max_new_tokens=gen_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # Benchmark
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                with torch.no_grad():
                    _ = self.model.generate(
                        **inputs,
                        max_new_tokens=gen_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                
                mem_after = torch.cuda.memory_allocated() / 1e9
                mem_peak = torch.cuda.max_memory_allocated() / 1e9
                
                total_tokens = batch_size * gen_tokens
                throughput = total_tokens / elapsed
                
                result = {
                    'batch_size': batch_size,
                    'time': elapsed,
                    'throughput': throughput,
                    'total_tokens': total_tokens,
                    'mem_before_gb': mem_before,
                    'mem_after_gb': mem_after,
                    'mem_peak_gb': mem_peak,
                }
                results.append(result)
                
                print(f"  Batch {batch_size}: {throughput:.1f} tokens/sec "
                      f"({elapsed:.3f}s for {total_tokens} tokens) "
                      f"[Peak mem: {mem_peak:.2f} GB]")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  Batch {batch_size}: ❌ OOM - not enough GPU memory")
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    raise
        
        # Calculate scaling efficiency
        if len(results) > 1:
            baseline = results[0]['throughput']
            for r in results[1:]:
                scaling = r['throughput'] / baseline
                ideal_scaling = r['batch_size'] / results[0]['batch_size']
                efficiency = (scaling / ideal_scaling) * 100
                print(f"\n  Batch {r['batch_size']} scaling: {scaling:.2f}x "
                      f"(ideal: {ideal_scaling}x, efficiency: {efficiency:.1f}%)")
        
        return results


def run_quick_benchmark(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    prompt_lengths: List[int] = [64, 128, 256],
    gen_tokens: int = 32,
) -> List[BenchmarkResult]:
    """Run a quick benchmark with default settings."""
    benchmark = InferenceBenchmark(model_name=model_name, load_in_4bit=True)
    benchmark.load()
    return benchmark.run(prompt_lengths=prompt_lengths, gen_tokens=gen_tokens)
