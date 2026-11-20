import torch

from decovton.attn_processor import AttnProcessor2_0, SkipAttnProcessor

def init_adapter(
    unet,
    cross_attn_cls=SkipAttnProcessor,
    self_attn_cls=None,
    cross_attn_dim=None,
    **kwargs,
):
    """
    Replace attention processors in the UNet with custom classes.

    By default, cross-attention blocks are replaced by SkipAttnProcessor
    (effectively disabling cross-attention), while self-attention blocks
    use AttnProcessor2_0.
    """
    if cross_attn_dim is None:
        cross_attn_dim = unet.config.cross_attention_dim

    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else cross_attn_dim

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            print(f"[WARNING] Unknown attention block name: {name}, skipping.")
            continue

        if cross_attention_dim is None:
            # self-attention
            if self_attn_cls is not None:
                attn_procs[name] = self_attn_cls(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    **kwargs,
                )
            else:
                attn_procs[name] = AttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    **kwargs,
                )
        else:
            # cross-attention
            attn_procs[name] = cross_attn_cls(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                **kwargs,
            )

    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    return adapter_modules
