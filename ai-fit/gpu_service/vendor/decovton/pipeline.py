import inspect
from typing import Union, Optional, List, Tuple
from PIL import Image

import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor

class DeCoVTONPipeline(DiffusionPipeline):
    """
    Diffusers-style pipeline for DeCo-VTON (Decouple, Don't Denoise, ECCV 2026).

    Inputs:
      - condition_image: (B,3,2H,W) RGB tensor in [-1,1], where the top half
        is the person image and the bottom half is the garment image.
      - mask: (B,1,H,W) binary mask indicating the person region to inpaint.
      - warped_garment: (B,3,H,W) optional warped garment image.

    The forward pass follows the analysis in our paper:
      - top latents are denoised via UNet,
      - bottom latents are constructed from garment latents + shared noise
        (with optional lookahead).
    """
    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: Union[
            DDPMScheduler,
            DDIMScheduler,
            UniPCMultistepScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    @classmethod
    def from_vton_checkpoint(
        cls,
        hf_repo: Optional[str] = None,
        subfolder: Optional[str] = None,
        base_model_path: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float32,
        vae_repo: str = "stabilityai/sd-vae-ft-mse",
        scheduler_repo: str = "stable-diffusion-v1-5/stable-diffusion-inpainting",
        local_files_only: bool = False,
    ) -> "DeCoVTONPipeline":
        """Assemble a DeCo-VTON pipeline from a fine-tuned UNet checkpoint.

        The released checkpoints contain only the UNet; the VAE and scheduler
        config are pulled from their original sources.
        """
        from decovton.attn_processor import SkipAttnProcessor
        from decovton.utils import init_adapter

        if (hf_repo is None) == (base_model_path is None):
            raise ValueError("Give exactly one of hf_repo or base_model_path.")

        vae = AutoencoderKL.from_pretrained(
            vae_repo, torch_dtype=torch_dtype, local_files_only=local_files_only
        )
        if hf_repo is not None:
            unet = UNet2DConditionModel.from_pretrained(
                hf_repo, subfolder=subfolder, torch_dtype=torch_dtype,
                local_files_only=local_files_only,
            )
        else:
            unet = UNet2DConditionModel.from_pretrained(
                base_model_path, subfolder="unet", torch_dtype=torch_dtype,
                local_files_only=local_files_only,
            )
        scheduler = DDPMScheduler.from_pretrained(
            scheduler_repo, subfolder="scheduler", local_files_only=local_files_only
        )

        # DeCo-VTON disables all cross-attention (text conditioning unused)
        init_adapter(unet, cross_attn_cls=SkipAttnProcessor)

        return cls(vae=vae, unet=unet, scheduler=scheduler)

    def enable_memory_efficient_mode(self):
        """Reduce VAE peak memory for 1024-resolution inference."""
        self.vae.enable_tiling()
        self.vae.enable_slicing()
        return self

    def prepare_extra_step_kwargs(self, generator=None, eta=None):
        """
        Prepare extra kwargs for scheduler.step (e.g., generator, eta).
        """
        extra = {}
        params = set(inspect.signature(self.scheduler.step).parameters.keys())
        if "generator" in params and generator is not None:
            extra["generator"] = generator
        if isinstance(self.scheduler, DDIMScheduler) and "eta" in params and eta is not None:
            extra["eta"] = float(eta)
        return extra

    def _maybe_swap_scheduler(self, scheduler_type: Optional[str] = None):
        """
        Optionally swap the internal scheduler at inference time.

        scheduler_type:
          - "ddim"
          - "ddpm"
          - "unipc"
          - "dpmpp"  (DPM-Solver++ multistep)
        """
        if scheduler_type is None:
            return

        s_type = scheduler_type.lower()
        if s_type == "ddim" and not isinstance(self.scheduler, DDIMScheduler):
            self.scheduler = DDIMScheduler.from_config(self.scheduler.config)
        elif s_type == "ddpm" and not isinstance(self.scheduler, DDPMScheduler):
            self.scheduler = DDPMScheduler.from_config(self.scheduler.config)
        elif s_type in ["unipc", "unipc_multistep"] and not isinstance(
            self.scheduler, UniPCMultistepScheduler
        ):
            self.scheduler = UniPCMultistepScheduler.from_config(self.scheduler.config)
        elif s_type in ["dpmpp", "dpm-solver++", "dpm_solver++", "dpm_solver"] and not isinstance(
            self.scheduler, DPMSolverMultistepScheduler
        ):
            self.scheduler = DPMSolverMultistepScheduler.from_config(
                self.scheduler.config
            )

    # optional CFG rescaling (not used in main experiments)
    def rescale_noise_cfg(self, noise_cfg, noise_pred_cond, guidance_rescale=0.0):
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_cond = noise_pred_cond.std(
            dim=list(range(1, noise_pred_cond.ndim)), keepdim=True
        )
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_cond / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = (
            guidance_rescale * noise_pred_rescaled +
            (1 - guidance_rescale) * noise_cfg
        )
        return noise_cfg

    @torch.no_grad()
    def __call__(
        self,
        condition_image: torch.FloatTensor,   # (B,3,2H,W), [top: person, bottom: cloth]
        mask: torch.FloatTensor,              # (B,1,H,W)
        warped_garment: Optional[torch.FloatTensor] = None,  # (B,3,H,W)
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        eta: Optional[float] = None,
        scheduler_type: Optional[str] = None,
        concat_mode: str = "latent",
        use_lookahead_garment: bool = False,
        lookahead_steps: int = 0,
        guidance_rescale: float = 0.0,
        **kwargs,
     ) -> Tuple[List[Image.Image], List[Image.Image]]:
        if concat_mode == "pixel" and warped_garment is not None:
            raise ValueError(
                "warped_garment is only supported with concat_mode='latent'; "
                "pixel concatenation is a legacy experimental path."
            )

        device = self._execution_device
        do_cfg = guidance_scale is not None and guidance_scale > 1.0
        concat_dim_spatial = 2  # concatenate/split along height

        # (0) Swap scheduler if requested
        self._maybe_swap_scheduler(scheduler_type)

        # (1) Prepare VAE dtype/device
        vae_dtype = self.unet.dtype
        self.vae.to(device=device, dtype=vae_dtype)

        # (2) Split condition_image into person/garment
        B, C, H2, W = condition_image.shape
        assert H2 % 2 == 0, "condition_image height must be 2H."
        H = H2 // 2

        person_img = condition_image[:, :, :H, :].to(device, dtype=vae_dtype)
        garment_img = condition_image[:, :, H:, :].to(device, dtype=vae_dtype)

        if concat_mode == "pixel":
            # Encode concatenated image and then split in latent space
            enc = self.vae.encode(condition_image.to(device, dtype=vae_dtype)).latent_dist
            lat = enc.sample() * self.vae.config.scaling_factor
            person_latent, garment_latent_0 = torch.chunk(lat, 2, dim=concat_dim_spatial)
        elif concat_mode == "latent":
            # Encode top/bottom separately
            person_latent = self.vae.encode(person_img).latent_dist.sample()
            garment_latent_0 = self.vae.encode(garment_img).latent_dist.sample()
            person_latent *= self.vae.config.scaling_factor
            garment_latent_0 *= self.vae.config.scaling_factor

            if warped_garment is not None:
                warped_latent = self.vae.encode(
                    warped_garment.to(device, dtype=vae_dtype)
                ).latent_dist.sample()
                warped_latent *= self.vae.config.scaling_factor
        else:
            raise ValueError("concat_mode must be 'pixel' or 'latent'.")

        latent_h, latent_w = person_latent.shape[2], person_latent.shape[3]

        # (3) Upsample mask to latent resolution (top half only; bottom is zeroed)
        mask_latent_top = torch.nn.functional.interpolate(
            mask.to(device, dtype=vae_dtype),
            size=(latent_h, latent_w),
            mode="nearest",
        )
        mask_latent = torch.cat(
            [mask_latent_top, torch.zeros_like(mask_latent_top)],
            dim=concat_dim_spatial,
        )  # (B,1,2h,w)

        # (4) Construct conditional / unconditional latents
        # cond: [person, z0_g], uncond: [person, 0]
        cond_latent_cond = torch.cat(
            [person_latent, garment_latent_0], dim=concat_dim_spatial
        )
        cond_latent_uncond = torch.cat(
            [person_latent, torch.zeros_like(garment_latent_0)],
            dim=concat_dim_spatial,
        )

        if do_cfg:
            cond_batch = torch.cat([cond_latent_uncond, cond_latent_cond], dim=0)
            mask_batch = torch.cat([mask_latent, mask_latent], dim=0)
        else:
            cond_batch = cond_latent_cond
            mask_batch = mask_latent

        # (5) Scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Initial noise for top latents
        latents_top = randn_tensor(
            person_latent.shape,
            generator=generator,
            device=device,
            dtype=vae_dtype,
        )
        latents_top = latents_top * self.scheduler.init_noise_sigma

        # Shared garment noise eps_g
        eps_g = randn_tensor(
            garment_latent_0.shape,
            generator=generator,
            device=device,
            dtype=vae_dtype,
        )

        # Precompute alphas_cumprod
        alphas_cumprod = self.scheduler.alphas_cumprod.to(
            device=device, dtype=vae_dtype
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator=generator, eta=eta)

        # (6) Denoising loop over timesteps
        for i, t in enumerate(self.progress_bar(timesteps)):
            # Current alpha_bar at timestep t
            alpha_bar_t = alphas_cumprod[t]
            sqrt_ab = alpha_bar_t.sqrt().view(1, 1, 1, 1)
            sqrt_1mab = (1.0 - alpha_bar_t).sqrt().view(1, 1, 1, 1)

            # Determine base latent for garment
            base_latent = warped_latent if warped_garment is not None else garment_latent_0

            # Compute garment latent at appropriate timestep
            if use_lookahead_garment: # Experimental: lookahead garment
                # Use lookahead timestep for cleaner garment signal
                idx = min(i + lookahead_steps, len(timesteps) - 1)
                t_look = timesteps[idx]
                alpha_bar_look = alphas_cumprod[t_look]
                sqrt_ab_look = alpha_bar_look.sqrt().view(1, 1, 1, 1)
                sqrt_1mab_look = (1.0 - alpha_bar_look).sqrt().view(1, 1, 1, 1)
                z_g_for_input = sqrt_ab_look * base_latent + sqrt_1mab_look * eps_g
            else:
                # Standard: use current timestep (Main result)
                z_g_for_input = sqrt_ab * base_latent + sqrt_1mab * eps_g

            # Construct full latents for cond / uncond
            latents_cond = torch.cat(
                [latents_top, z_g_for_input], dim=concat_dim_spatial
            )
            latents_uncond = torch.cat(
                [latents_top, torch.zeros_like(z_g_for_input)],
                dim=concat_dim_spatial,
            )

            if do_cfg:
                latents_model_input = torch.cat(
                    [latents_uncond, latents_cond], dim=0
                )
            else:
                latents_model_input = latents_cond

            # Scheduler scaling
            latents_model_input = self.scheduler.scale_model_input(
                latents_model_input, t
            )

            # 9-channel UNet input: [z_t | mask | cond_latent]
            model_input = torch.cat(
                [latents_model_input, mask_batch, cond_batch],
                dim=1,
            )  # (B or 2B, 4+1+4, 2h,w)

            # UNet forward
            noise_pred = self.unet(
                model_input,
                t,
                encoder_hidden_states=None,
                return_dict=False,
            )[0]

            # CFG
            if do_cfg:
                noise_uncond, noise_cond = noise_pred.chunk(2, dim=0)
                noise_pred_cfg = noise_uncond + guidance_scale * (
                    noise_cond - noise_uncond
                )

                if guidance_rescale > 0.0:
                    # Rescale based on top-half statistics
                    nu_top, _ = noise_uncond.chunk(2, dim=2)
                    nc_top, _ = noise_cond.chunk(2, dim=2)
                    std_cond = nc_top.std(
                        dim=list(range(1, nc_top.ndim)), keepdim=True
                    )
                    std_cfg = (
                        noise_pred_cfg.chunk(2, dim=2)[0]
                        .std(dim=list(range(1, nc_top.ndim)), keepdim=True)
                    )
                    noise_pred_rescaled = noise_pred_cfg * (std_cond / (std_cfg + 1e-8))
                    noise_pred_cfg = (
                        guidance_rescale * noise_pred_rescaled
                        + (1 - guidance_rescale) * noise_pred_cfg
                    )

                noise_pred = noise_pred_cfg

            # Scheduler step (use cond branch latents)
            step = self.scheduler.step(
                noise_pred,
                t,
                latents_cond,
                **extra_step_kwargs,
            )
            prev = step.prev_sample if hasattr(step, "prev_sample") else step[0]

            # Keep only the top half for the next iteration
            latents_top, _ = torch.chunk(
                prev, 2, dim=concat_dim_spatial
            )

        # (7) Decode top and bottom latents
        top_latents = latents_top / self.vae.config.scaling_factor
        image_top = self.vae.decode(top_latents).sample  # [-1,1]

        bottom_latents = z_g_for_input / self.vae.config.scaling_factor
        image_bottom = self.vae.decode(bottom_latents).sample  # [-1,1]

        image_top = (image_top / 2 + 0.5).clamp(0, 1)
        image_bottom = (image_bottom / 2 + 0.5).clamp(0, 1)

        if output_type == "pil":
            top_np = image_top.detach().to(torch.float32).cpu().permute(0, 2, 3, 1).numpy()
            bot_np = image_bottom.detach().to(torch.float32).cpu().permute(0, 2, 3, 1).numpy()
            top_pil = self.numpy_to_pil(top_np)
            bot_pil = self.numpy_to_pil(bot_np)
            return top_pil, bot_pil
        else:
            top_np = (image_top * 255).round().to(torch.uint8).detach().cpu().permute(0, 2, 3, 1).numpy()
            bot_np = (image_bottom * 255).round().to(torch.uint8).detach().cpu().permute(0, 2, 3, 1).numpy()
            return top_np, bot_np


# Backward-compat alias for the Re-CatVTON release
RECATVTONPipeline = DeCoVTONPipeline
