from typing import Optional
import torch
from torch.nn import functional as F

class SkipAttnProcessor(torch.nn.Module):
    """
    Attention processor that simply returns the input hidden_states.

    Used to disable cross-attention in DeCo-VTON, effectively turning the
    UNet into a purely image-conditional model (no text encoder).
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ):
        return hidden_states

class AttnProcessor2_0(torch.nn.Module):
    """
    Processor for implementing scaled dot-product attention
    (enabled by default if you're using PyTorch 2.0).

    This is essentially a light wrapper around F.scaled_dot_product_attention
    with the same interface as Diffusers' attention processors.
    """
    def __init__(
        self,
        hidden_size: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0; please upgrade PyTorch "
                "to 2.0 or higher."
            )

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        residual = hidden_states
        input_ndim = hidden_states.ndim

        # spatial norm (for 2D feature maps)
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if input_ndim == 4:
            # (B, C, H, W) -> (B, HW, C)
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            batch_size, sequence_length, _ = hidden_states.shape

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        # projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, key.shape[2], batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        # scaled dot-product attention
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)  # dropout

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
