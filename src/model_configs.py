"""
LLM Architecture Configurations for FLOP Calculations

Contains architecture parameters for common LLMs used to
calculate theoretical FLOPs for roofline analysis.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Base LLM architecture configuration."""
    
    name: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int  # For GQA, equals num_attention_heads for MHA
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    
    @property
    def params_billions(self) -> float:
        """Approximate parameter count in billions."""
        # Embedding
        embed = self.vocab_size * self.hidden_size
        
        # Per layer - Attention
        q_proj = self.hidden_size * self.hidden_size
        k_proj = self.hidden_size * (self.num_key_value_heads * self.head_dim)
        v_proj = self.hidden_size * (self.num_key_value_heads * self.head_dim)
        o_proj = self.hidden_size * self.hidden_size
        attn_per_layer = q_proj + k_proj + v_proj + o_proj
        
        # Per layer - MLP (assuming SwiGLU with gate, up, down)
        mlp_per_layer = 3 * self.hidden_size * self.intermediate_size
        
        # Layer norms
        ln_per_layer = 2 * self.hidden_size
        
        total_per_layer = attn_per_layer + mlp_per_layer + ln_per_layer
        
        # LM head (often tied with embedding)
        lm_head = self.vocab_size * self.hidden_size
        
        total = embed + (self.num_hidden_layers * total_per_layer) + lm_head
        return total / 1e9
    
    @property
    def uses_gqa(self) -> bool:
        """Check if model uses Grouped Query Attention."""
        return self.num_key_value_heads < self.num_attention_heads
    
    @property
    def gqa_ratio(self) -> int:
        """Ratio of query heads to KV heads."""
        return self.num_attention_heads // self.num_key_value_heads
    
    def __str__(self) -> str:
        gqa_str = f" (GQA {self.gqa_ratio}:1)" if self.uses_gqa else " (MHA)"
        return (
            f"{self.name}\n"
            f"  Parameters: ~{self.params_billions:.2f}B\n"
            f"  Hidden size: {self.hidden_size}\n"
            f"  Layers: {self.num_hidden_layers}\n"
            f"  Attention: {self.num_attention_heads} heads{gqa_str}\n"
            f"  Vocab size: {self.vocab_size}"
        )


# Pre-defined model configurations

class MistralConfig(ModelConfig):
    """Mistral-7B architecture."""
    def __init__(self):
        super().__init__(
            name="Mistral-7B",
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,  # GQA
            head_dim=128,
            vocab_size=32000,
            max_position_embeddings=32768,
        )


class Llama2_7BConfig(ModelConfig):
    """Llama-2-7B architecture."""
    def __init__(self):
        super().__init__(
            name="Llama-2-7B",
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,  # MHA
            head_dim=128,
            vocab_size=32000,
            max_position_embeddings=4096,
        )


class Llama2_13BConfig(ModelConfig):
    """Llama-2-13B architecture."""
    def __init__(self):
        super().__init__(
            name="Llama-2-13B",
            hidden_size=5120,
            intermediate_size=13824,
            num_hidden_layers=40,
            num_attention_heads=40,
            num_key_value_heads=40,  # MHA
            head_dim=128,
            vocab_size=32000,
            max_position_embeddings=4096,
        )


class Llama2_70BConfig(ModelConfig):
    """Llama-2-70B architecture."""
    def __init__(self):
        super().__init__(
            name="Llama-2-70B",
            hidden_size=8192,
            intermediate_size=28672,
            num_hidden_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,  # GQA
            head_dim=128,
            vocab_size=32000,
            max_position_embeddings=4096,
        )


class Llama3_8BConfig(ModelConfig):
    """Llama-3-8B architecture."""
    def __init__(self):
        super().__init__(
            name="Llama-3-8B",
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,  # GQA
            head_dim=128,
            vocab_size=128256,
            max_position_embeddings=8192,
        )


class Llama3_70BConfig(ModelConfig):
    """Llama-3-70B architecture."""
    def __init__(self):
        super().__init__(
            name="Llama-3-70B",
            hidden_size=8192,
            intermediate_size=28672,
            num_hidden_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,  # GQA
            head_dim=128,
            vocab_size=128256,
            max_position_embeddings=8192,
        )


class Phi2Config(ModelConfig):
    """Phi-2 (2.7B) architecture."""
    def __init__(self):
        super().__init__(
            name="Phi-2",
            hidden_size=2560,
            intermediate_size=10240,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,  # MHA
            head_dim=80,
            vocab_size=51200,
            max_position_embeddings=2048,
        )


class Gemma7BConfig(ModelConfig):
    """Gemma-7B architecture."""
    def __init__(self):
        super().__init__(
            name="Gemma-7B",
            hidden_size=3072,
            intermediate_size=24576,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=16,  # MHA
            head_dim=256,
            vocab_size=256000,
            max_position_embeddings=8192,
        )


class Qwen2_7BConfig(ModelConfig):
    """Qwen2-7B architecture."""
    def __init__(self):
        super().__init__(
            name="Qwen2-7B",
            hidden_size=3584,
            intermediate_size=18944,
            num_hidden_layers=28,
            num_attention_heads=28,
            num_key_value_heads=4,  # GQA
            head_dim=128,
            vocab_size=152064,
            max_position_embeddings=32768,
        )


def create_custom_config(
    name: str,
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    vocab_size: int = 32000,
    max_seq_len: int = 4096,
) -> ModelConfig:
    """Create a custom model configuration."""
    if num_kv_heads is None:
        num_kv_heads = num_heads
    if head_dim is None:
        head_dim = hidden_size // num_heads
    
    return ModelConfig(
        name=name,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        max_position_embeddings=max_seq_len,
    )


# Convenience dict for all models
ALL_MODELS = {
    "mistral-7b": MistralConfig,
    "llama2-7b": Llama2_7BConfig,
    "llama2-13b": Llama2_13BConfig,
    "llama2-70b": Llama2_70BConfig,
    "llama3-8b": Llama3_8BConfig,
    "llama3-70b": Llama3_70BConfig,
    "phi2": Phi2Config,
    "gemma-7b": Gemma7BConfig,
    "qwen2-7b": Qwen2_7BConfig,
}
