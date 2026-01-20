"""
GPU Hardware Specifications for Roofline Analysis

Contains theoretical peak performance and memory bandwidth
for common GPUs used in LLM inference.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUSpecs:
    """Base GPU specifications for roofline analysis."""
    
    name: str
    peak_fp32_tflops: float
    peak_fp16_tflops: float
    peak_int8_tops: float
    memory_bandwidth_gb_s: float
    memory_gb: float
    tdp_watts: float
    practical_efficiency: float = 0.7  # Typical achievable % of peak
    
    @property
    def ridge_point_fp16(self) -> float:
        """Operational intensity where compute meets memory bandwidth (FP16)."""
        return (self.peak_fp16_tflops * 1e12) / (self.memory_bandwidth_gb_s * 1e9)
    
    @property
    def ridge_point_fp32(self) -> float:
        """Operational intensity where compute meets memory bandwidth (FP32)."""
        return (self.peak_fp32_tflops * 1e12) / (self.memory_bandwidth_gb_s * 1e9)
    
    @property
    def ridge_point_int8(self) -> float:
        """Operational intensity where compute meets memory bandwidth (INT8)."""
        return (self.peak_int8_tops * 1e12) / (self.memory_bandwidth_gb_s * 1e9)
    
    def practical_peak_fp16(self) -> float:
        """Realistically achievable FP16 performance."""
        return self.peak_fp16_tflops * self.practical_efficiency
    
    def is_memory_bound(self, operational_intensity: float, dtype: str = "fp16") -> bool:
        """Check if a workload with given OI is memory-bound."""
        ridge = {
            "fp32": self.ridge_point_fp32,
            "fp16": self.ridge_point_fp16,
            "int8": self.ridge_point_int8,
        }.get(dtype, self.ridge_point_fp16)
        return operational_intensity < ridge
    
    def theoretical_performance(self, operational_intensity: float, dtype: str = "fp16") -> float:
        """
        Calculate theoretical max performance at given operational intensity.
        
        Returns: Performance in TFLOPS
        """
        peak = {
            "fp32": self.peak_fp32_tflops,
            "fp16": self.peak_fp16_tflops,
            "int8": self.peak_int8_tops,
        }.get(dtype, self.peak_fp16_tflops)
        
        # Memory-bound: performance = bandwidth * OI
        memory_limited = (self.memory_bandwidth_gb_s * operational_intensity) / 1000
        
        # Compute-bound: performance = peak
        return min(memory_limited, peak)
    
    def __str__(self) -> str:
        return (
            f"{self.name}\n"
            f"  Peak FP16: {self.peak_fp16_tflops} TFLOPS\n"
            f"  Peak FP32: {self.peak_fp32_tflops} TFLOPS\n"
            f"  Memory BW: {self.memory_bandwidth_gb_s} GB/s\n"
            f"  Memory: {self.memory_gb} GB\n"
            f"  Ridge Point (FP16): {self.ridge_point_fp16:.1f} FLOPs/Byte"
        )


# Pre-defined GPU specifications

class T4Specs(GPUSpecs):
    """NVIDIA T4 - Common in Colab free tier."""
    def __init__(self):
        super().__init__(
            name="NVIDIA T4",
            peak_fp32_tflops=8.1,
            peak_fp16_tflops=65.0,
            peak_int8_tops=130.0,
            memory_bandwidth_gb_s=320.0,
            memory_gb=16.0,
            tdp_watts=70.0,
        )


class V100Specs(GPUSpecs):
    """NVIDIA V100 16GB."""
    def __init__(self):
        super().__init__(
            name="NVIDIA V100",
            peak_fp32_tflops=15.7,
            peak_fp16_tflops=125.0,
            peak_int8_tops=250.0,
            memory_bandwidth_gb_s=900.0,
            memory_gb=16.0,
            tdp_watts=300.0,
        )


class A100_40GBSpecs(GPUSpecs):
    """NVIDIA A100 40GB."""
    def __init__(self):
        super().__init__(
            name="NVIDIA A100 40GB",
            peak_fp32_tflops=19.5,
            peak_fp16_tflops=312.0,
            peak_int8_tops=624.0,
            memory_bandwidth_gb_s=1555.0,
            memory_gb=40.0,
            tdp_watts=400.0,
        )


class A100_80GBSpecs(GPUSpecs):
    """NVIDIA A100 80GB."""
    def __init__(self):
        super().__init__(
            name="NVIDIA A100 80GB",
            peak_fp32_tflops=19.5,
            peak_fp16_tflops=312.0,
            peak_int8_tops=624.0,
            memory_bandwidth_gb_s=2039.0,
            memory_gb=80.0,
            tdp_watts=400.0,
        )


class H100Specs(GPUSpecs):
    """NVIDIA H100 SXM."""
    def __init__(self):
        super().__init__(
            name="NVIDIA H100",
            peak_fp32_tflops=67.0,
            peak_fp16_tflops=990.0,
            peak_int8_tops=1980.0,
            memory_bandwidth_gb_s=3350.0,
            memory_gb=80.0,
            tdp_watts=700.0,
        )


class L4Specs(GPUSpecs):
    """NVIDIA L4 - Available in Colab Pro."""
    def __init__(self):
        super().__init__(
            name="NVIDIA L4",
            peak_fp32_tflops=30.3,
            peak_fp16_tflops=121.0,
            peak_int8_tops=242.0,
            memory_bandwidth_gb_s=300.0,
            memory_gb=24.0,
            tdp_watts=72.0,
        )


class RTX4090Specs(GPUSpecs):
    """NVIDIA RTX 4090 - Consumer GPU."""
    def __init__(self):
        super().__init__(
            name="NVIDIA RTX 4090",
            peak_fp32_tflops=82.6,
            peak_fp16_tflops=165.2,
            peak_int8_tops=330.4,
            memory_bandwidth_gb_s=1008.0,
            memory_gb=24.0,
            tdp_watts=450.0,
        )


# GPU detection utility
def detect_gpu() -> Optional[GPUSpecs]:
    """Auto-detect GPU and return appropriate specs."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        
        gpu_name = torch.cuda.get_device_name(0)
        
        gpu_mapping = {
            "T4": T4Specs,
            "V100": V100Specs,
            "A100-SXM4-40GB": A100_40GBSpecs,
            "A100-SXM4-80GB": A100_80GBSpecs,
            "A100 40GB": A100_40GBSpecs,
            "A100 80GB": A100_80GBSpecs,
            "H100": H100Specs,
            "L4": L4Specs,
            "RTX 4090": RTX4090Specs,
        }
        
        for key, specs_class in gpu_mapping.items():
            if key in gpu_name:
                return specs_class()
        
        print(f"Unknown GPU: {gpu_name}, defaulting to T4 specs")
        return T4Specs()
        
    except ImportError:
        print("PyTorch not available for GPU detection")
        return None


# Convenience dict for all GPUs
ALL_GPUS = {
    "T4": T4Specs,
    "V100": V100Specs,
    "A100-40GB": A100_40GBSpecs,
    "A100-80GB": A100_80GBSpecs,
    "H100": H100Specs,
    "L4": L4Specs,
    "RTX4090": RTX4090Specs,
}
