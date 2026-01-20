"""
Roofline Analyzer - LLM Inference Performance Analysis

A toolkit for roofline analysis of LLM inference workloads.
"""

from .gpu_specs import (
    GPUSpecs,
    T4Specs,
    V100Specs,
    A100_40GBSpecs,
    A100_80GBSpecs,
    H100Specs,
    L4Specs,
    RTX4090Specs,
    detect_gpu,
    ALL_GPUS,
)

from .model_configs import (
    ModelConfig,
    MistralConfig,
    Llama2_7BConfig,
    Llama2_13BConfig,
    Llama2_70BConfig,
    Llama3_8BConfig,
    Llama3_70BConfig,
    Phi2Config,
    Gemma7BConfig,
    Qwen2_7BConfig,
    create_custom_config,
    ALL_MODELS,
)

from .roofline import (
    RooflineAnalyzer,
    FLOPBreakdown,
    MemoryBreakdown,
    AnalysisResult,
    quick_analysis,
)

__version__ = "0.1.0"

__all__ = [
    # GPU Specs
    "GPUSpecs",
    "T4Specs",
    "V100Specs", 
    "A100_40GBSpecs",
    "A100_80GBSpecs",
    "H100Specs",
    "L4Specs",
    "RTX4090Specs",
    "detect_gpu",
    "ALL_GPUS",
    # Model Configs
    "ModelConfig",
    "MistralConfig",
    "Llama2_7BConfig",
    "Llama2_13BConfig",
    "Llama2_70BConfig",
    "Llama3_8BConfig",
    "Llama3_70BConfig",
    "Phi2Config",
    "Gemma7BConfig",
    "Qwen2_7BConfig",
    "create_custom_config",
    "ALL_MODELS",
    # Roofline Analysis
    "RooflineAnalyzer",
    "FLOPBreakdown",
    "MemoryBreakdown",
    "AnalysisResult",
    "quick_analysis",
]
