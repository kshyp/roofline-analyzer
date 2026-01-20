"""
Roofline Analysis for LLM Inference

Core module for calculating operational intensity, theoretical FLOPs,
and visualizing roofline plots for transformer inference workloads.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from .gpu_specs import GPUSpecs, T4Specs, detect_gpu
from .model_configs import ModelConfig, MistralConfig


@dataclass
class FLOPBreakdown:
    """Breakdown of FLOPs by component."""
    q_proj: float
    k_proj: float
    v_proj: float
    attn_scores: float
    attn_output: float
    o_proj: float
    mlp_gate: float
    mlp_up: float
    mlp_down: float
    lm_head: float
    total: float
    
    def as_dict(self) -> Dict[str, float]:
        return {
            "q_proj": self.q_proj,
            "k_proj": self.k_proj,
            "v_proj": self.v_proj,
            "attn_scores": self.attn_scores,
            "attn_output": self.attn_output,
            "o_proj": self.o_proj,
            "mlp_gate": self.mlp_gate,
            "mlp_up": self.mlp_up,
            "mlp_down": self.mlp_down,
            "lm_head": self.lm_head,
            "total": self.total,
        }


@dataclass 
class MemoryBreakdown:
    """Breakdown of memory traffic by component."""
    weights: float
    kv_cache: float
    activations: float
    total: float


@dataclass
class AnalysisResult:
    """Result of roofline analysis for a workload."""
    phase: str
    seq_len: int
    batch_size: int
    flops: float
    memory_bytes: float
    operational_intensity: float
    theoretical_tflops: float
    is_memory_bound: bool
    bottleneck: str
    efficiency_ceiling: float  # Max achievable % of peak


class RooflineAnalyzer:
    """
    Roofline analyzer for LLM inference workloads.
    
    Example:
        analyzer = RooflineAnalyzer(
            gpu_specs=T4Specs(),
            model_config=MistralConfig()
        )
        
        # Analyze prefill phase
        result = analyzer.analyze(seq_len=512, phase="prefill")
        print(f"Bottleneck: {result.bottleneck}")
        
        # Plot roofline
        analyzer.plot_roofline(save_path="roofline.png")
    """
    
    def __init__(
        self,
        gpu_specs: Optional[GPUSpecs] = None,
        model_config: Optional[ModelConfig] = None,
    ):
        self.gpu = gpu_specs or detect_gpu() or T4Specs()
        self.model = model_config or MistralConfig()
    
    def calculate_prefill_flops(
        self,
        seq_len: int,
        batch_size: int = 1,
    ) -> FLOPBreakdown:
        """
        Calculate FLOPs for prefill phase (processing input prompt).
        
        Prefill processes all tokens in parallel, so attention is O(n²).
        """
        B, S, H = batch_size, seq_len, self.model.hidden_size
        L = self.model.num_hidden_layers
        I = self.model.intermediate_size
        V = self.model.vocab_size
        kv_heads = self.model.num_key_value_heads
        head_dim = self.model.head_dim
        n_heads = self.model.num_attention_heads
        kv_dim = kv_heads * head_dim
        
        # All FLOPs multiplied by 2 for multiply-add
        q_proj = 2 * B * S * H * H * L
        k_proj = 2 * B * S * H * kv_dim * L
        v_proj = 2 * B * S * H * kv_dim * L
        
        # Attention: Q @ K^T and scores @ V
        attn_scores = 2 * B * n_heads * S * S * head_dim * L
        attn_output = 2 * B * n_heads * S * S * head_dim * L
        
        o_proj = 2 * B * S * H * H * L
        
        # MLP (SwiGLU: gate, up, down)
        mlp_gate = 2 * B * S * H * I * L
        mlp_up = 2 * B * S * H * I * L
        mlp_down = 2 * B * S * I * H * L
        
        # LM head
        lm_head = 2 * B * S * H * V
        
        total = (q_proj + k_proj + v_proj + attn_scores + attn_output +
                 o_proj + mlp_gate + mlp_up + mlp_down + lm_head)
        
        return FLOPBreakdown(
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            attn_scores=attn_scores,
            attn_output=attn_output,
            o_proj=o_proj,
            mlp_gate=mlp_gate,
            mlp_up=mlp_up,
            mlp_down=mlp_down,
            lm_head=lm_head,
            total=total,
        )
    
    def calculate_decode_flops(
        self,
        kv_cache_len: int,
        batch_size: int = 1,
    ) -> FLOPBreakdown:
        """
        Calculate FLOPs for decode phase (generating one token).
        
        Decode generates one token at a time, attending to KV cache.
        """
        B, H = batch_size, self.model.hidden_size
        L = self.model.num_hidden_layers
        I = self.model.intermediate_size
        V = self.model.vocab_size
        kv_heads = self.model.num_key_value_heads
        head_dim = self.model.head_dim
        n_heads = self.model.num_attention_heads
        kv_dim = kv_heads * head_dim
        S = kv_cache_len  # Context length for attention
        
        # Single token projections
        q_proj = 2 * B * 1 * H * H * L
        k_proj = 2 * B * 1 * H * kv_dim * L
        v_proj = 2 * B * 1 * H * kv_dim * L
        
        # Attention with KV cache (1 query token attends to S cached tokens)
        attn_scores = 2 * B * n_heads * 1 * S * head_dim * L
        attn_output = 2 * B * n_heads * 1 * S * head_dim * L
        
        o_proj = 2 * B * 1 * H * H * L
        
        # MLP
        mlp_gate = 2 * B * 1 * H * I * L
        mlp_up = 2 * B * 1 * H * I * L
        mlp_down = 2 * B * 1 * I * H * L
        
        # LM head
        lm_head = 2 * B * 1 * H * V
        
        total = (q_proj + k_proj + v_proj + attn_scores + attn_output +
                 o_proj + mlp_gate + mlp_up + mlp_down + lm_head)
        
        return FLOPBreakdown(
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            attn_scores=attn_scores,
            attn_output=attn_output,
            o_proj=o_proj,
            mlp_gate=mlp_gate,
            mlp_up=mlp_up,
            mlp_down=mlp_down,
            lm_head=lm_head,
            total=total,
        )
    
    def calculate_memory_traffic(
        self,
        seq_len: int,
        batch_size: int = 1,
        dtype_bytes: int = 2,  # FP16 = 2 bytes
        phase: str = "prefill",
    ) -> MemoryBreakdown:
        """
        Estimate memory traffic (bytes read + written).
        
        For inference, main cost is reading model weights once per forward pass.
        """
        H = self.model.hidden_size
        L = self.model.num_hidden_layers
        I = self.model.intermediate_size
        V = self.model.vocab_size
        kv_heads = self.model.num_key_value_heads
        head_dim = self.model.head_dim
        kv_dim = kv_heads * head_dim
        
        # Weight reads (read once per layer)
        weights = 0
        weights += H * H * dtype_bytes * L  # Q proj
        weights += H * kv_dim * dtype_bytes * L  # K proj
        weights += H * kv_dim * dtype_bytes * L  # V proj
        weights += H * H * dtype_bytes * L  # O proj
        weights += H * I * dtype_bytes * L  # MLP gate
        weights += H * I * dtype_bytes * L  # MLP up
        weights += I * H * dtype_bytes * L  # MLP down
        weights += H * V * dtype_bytes  # LM head
        weights += V * H * dtype_bytes  # Embedding
        
        # KV cache and activations
        if phase == "prefill":
            kv_cache = 0  # No KV cache read during prefill
            activations = batch_size * seq_len * H * dtype_bytes * L * 4
        else:  # decode
            # Read entire KV cache
            kv_cache = 2 * batch_size * seq_len * kv_dim * dtype_bytes * L
            activations = batch_size * 1 * H * dtype_bytes * L * 4
        
        total = weights + kv_cache + activations
        
        return MemoryBreakdown(
            weights=weights,
            kv_cache=kv_cache,
            activations=activations,
            total=total,
        )
    
    def get_operational_intensity(
        self,
        seq_len: int,
        batch_size: int = 1,
        phase: str = "prefill",
    ) -> float:
        """
        Calculate operational intensity (FLOPs / Byte).
        
        Higher OI = more compute per memory access = more likely compute-bound.
        """
        if phase == "prefill":
            flops = self.calculate_prefill_flops(seq_len, batch_size).total
        else:
            flops = self.calculate_decode_flops(seq_len, batch_size).total
        
        memory = self.calculate_memory_traffic(seq_len, batch_size, phase=phase).total
        
        return flops / memory if memory > 0 else 0
    
    def analyze(
        self,
        seq_len: int,
        batch_size: int = 1,
        phase: str = "prefill",
    ) -> AnalysisResult:
        """
        Perform complete roofline analysis for a workload.
        """
        if phase == "prefill":
            flops = self.calculate_prefill_flops(seq_len, batch_size).total
        else:
            flops = self.calculate_decode_flops(seq_len, batch_size).total
        
        memory = self.calculate_memory_traffic(seq_len, batch_size, phase=phase).total
        oi = flops / memory if memory > 0 else 0
        
        is_memory_bound = self.gpu.is_memory_bound(oi)
        theoretical = self.gpu.theoretical_performance(oi)
        
        if is_memory_bound:
            bottleneck = "memory-bound"
            efficiency_ceiling = (oi / self.gpu.ridge_point_fp16) * 100
        else:
            bottleneck = "compute-bound"
            efficiency_ceiling = 100.0
        
        return AnalysisResult(
            phase=phase,
            seq_len=seq_len,
            batch_size=batch_size,
            flops=flops,
            memory_bytes=memory,
            operational_intensity=oi,
            theoretical_tflops=theoretical,
            is_memory_bound=is_memory_bound,
            bottleneck=bottleneck,
            efficiency_ceiling=min(efficiency_ceiling, 100.0),
        )
    
    def plot_roofline(
        self,
        measured_points: Optional[List[Dict]] = None,
        seq_lengths: List[int] = [128, 256, 512, 1024, 2048],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """
        Create a roofline plot.
        
        Args:
            measured_points: List of dicts with 'name', 'oi', 'tflops' keys
            seq_lengths: Sequence lengths to plot theoretical points
            save_path: Path to save figure (optional)
            figsize: Figure size
        
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Roofline parameters
        peak_fp16 = self.gpu.peak_fp16_tflops
        mem_bw = self.gpu.memory_bandwidth_gb_s
        ridge_point = self.gpu.ridge_point_fp16
        
        # Create roofline curve
        oi_range = np.logspace(-1, 4, 500)
        memory_bound = mem_bw * oi_range / 1000  # Convert to TFLOPS
        compute_bound = np.full_like(oi_range, peak_fp16)
        roofline = np.minimum(memory_bound, compute_bound)
        
        # Plot roofline
        ax.loglog(oi_range, roofline, 'b-', linewidth=3, 
                  label=f'{self.gpu.name} Roofline (FP16)')
        
        # Practical roofline (70% efficiency)
        practical = roofline * self.gpu.practical_efficiency
        ax.loglog(oi_range, practical, 'b--', linewidth=2, alpha=0.5,
                  label=f'Practical (~{int(self.gpu.practical_efficiency*100)}% efficiency)')
        
        # Ridge point
        ax.axvline(ridge_point, color='gray', linestyle=':', alpha=0.7)
        ax.annotate(f'Ridge Point\n({ridge_point:.0f} F/B)',
                    xy=(ridge_point, peak_fp16),
                    xytext=(ridge_point * 2, peak_fp16 * 0.7),
                    fontsize=10, ha='left',
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))
        
        # Plot theoretical prefill points
        prefill_colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(seq_lengths)))
        for i, seq_len in enumerate(seq_lengths):
            result = self.analyze(seq_len, phase="prefill")
            theoretical_perf = result.theoretical_tflops * 0.5  # Assume 50% of theoretical
            ax.scatter([result.operational_intensity], [theoretical_perf],
                       c=[prefill_colors[i]], s=100, marker='s',
                       edgecolors='black', linewidths=1, zorder=5)
            ax.annotate(f'Prefill {seq_len}',
                        xy=(result.operational_intensity, theoretical_perf),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot theoretical decode points
        decode_colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(seq_lengths)))
        for i, kv_len in enumerate(seq_lengths):
            result = self.analyze(kv_len, phase="decode")
            theoretical_perf = result.theoretical_tflops * 0.4
            ax.scatter([result.operational_intensity], [theoretical_perf],
                       c=[decode_colors[i]], s=100, marker='^',
                       edgecolors='black', linewidths=1, zorder=5)
            ax.annotate(f'Decode {kv_len}',
                        xy=(result.operational_intensity, theoretical_perf),
                        xytext=(5, -10), textcoords='offset points', fontsize=8)
        
        # Plot measured points
        if measured_points:
            for point in measured_points:
                ax.scatter([point['oi']], [point['tflops']],
                           c='red', s=200, marker='*',
                           edgecolors='darkred', linewidths=1.5, zorder=10,
                           label=f"Measured ({point.get('name', 'unknown')})")
        
        # Formatting
        ax.set_xlabel('Operational Intensity (FLOPs/Byte)', fontsize=12)
        ax.set_ylabel('Performance (TFLOPS)', fontsize=12)
        ax.set_title(f'Roofline Analysis: {self.model.name} on {self.gpu.name}',
                     fontsize=14, fontweight='bold')
        
        ax.set_xlim(0.1, 10000)
        ax.set_ylim(0.1, peak_fp16 * 2)
        
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
        
        # Region labels
        ax.text(0.3, 0.3, 'Memory\nBound', fontsize=14, alpha=0.5,
                ha='center', transform=ax.transAxes)
        ax.text(0.85, 0.7, 'Compute\nBound', fontsize=14, alpha=0.5,
                ha='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved roofline plot to {save_path}")
        
        return fig
    
    def summary_table(
        self,
        seq_lengths: List[int] = [128, 256, 512, 1024, 2048],
    ) -> str:
        """Generate a summary table of analysis results."""
        lines = [
            "=" * 80,
            f"Roofline Analysis: {self.model.name} on {self.gpu.name}",
            "=" * 80,
            f"{'Phase':<10} {'Seq Len':<10} {'FLOPs':<15} {'Memory':<12} {'OI (F/B)':<10} {'Bound':<12}",
            "-" * 80,
        ]
        
        for seq_len in seq_lengths:
            result = self.analyze(seq_len, phase="prefill")
            lines.append(
                f"{'Prefill':<10} {seq_len:<10} {result.flops/1e12:.2f} TFLOPS   "
                f"{result.memory_bytes/1e9:.2f} GB    {result.operational_intensity:.1f}        "
                f"{result.bottleneck}"
            )
        
        lines.append("-" * 80)
        
        for kv_len in seq_lengths:
            result = self.analyze(kv_len, phase="decode")
            lines.append(
                f"{'Decode':<10} {kv_len:<10} {result.flops/1e9:.2f} GFLOPS   "
                f"{result.memory_bytes/1e9:.2f} GB    {result.operational_intensity:.1f}        "
                f"{result.bottleneck}"
            )
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


def quick_analysis(
    model: str = "mistral-7b",
    gpu: str = "T4",
    seq_len: int = 512,
) -> None:
    """Quick roofline analysis with default settings."""
    from .gpu_specs import ALL_GPUS
    from .model_configs import ALL_MODELS
    
    gpu_specs = ALL_GPUS.get(gpu, T4Specs)()
    model_config = ALL_MODELS.get(model, MistralConfig)()
    
    analyzer = RooflineAnalyzer(gpu_specs, model_config)
    
    print(analyzer.summary_table())
    
    prefill = analyzer.analyze(seq_len, phase="prefill")
    decode = analyzer.analyze(seq_len, phase="decode")
    
    print(f"\nKey Insights for seq_len={seq_len}:")
    print(f"  Prefill: {prefill.bottleneck} (OI={prefill.operational_intensity:.1f})")
    print(f"  Decode: {decode.bottleneck} (OI={decode.operational_intensity:.1f})")
    print(f"  Ridge point: {gpu_specs.ridge_point_fp16:.1f} FLOPs/Byte")
