"""DeCo-VTON adapter with the local Stable Diffusion safety checker."""
import numpy as np
import torch
from PIL import Image, ImageOps


class OutputBlocked(Exception):
    pass


def _from_pretrained(loader, *args, **kwargs):
    try:
        return loader(*args, local_files_only=True, **kwargs)
    except (OSError, ValueError):
        return loader(*args, local_files_only=False, **kwargs)


class DeCoBackend:
    resolution = (384, 512)

    def resident_bytes(self):
        return sum(p.numel()*p.element_size() for model in (self.core.unet,self.core.vae) for p in model.parameters())

    def park_cpu(self):
        for model in (self.core.unet,self.core.vae):model.to('cpu',non_blocking=True)
        torch.cuda.synchronize()

    def activate_cuda(self):
        for model in (self.core.unet,self.core.vae):model.to(self.device,non_blocking=True)
        torch.cuda.synchronize()

    def __init__(self, checkpoint):
        from decovton import DeCoVTONPipeline
        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
        from transformers import CLIPImageProcessor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.resolution = (512,768)
        # Use the same VAE as the verified test and the released model.
        load = dict(base_model_path=str(checkpoint), torch_dtype=self.dtype,
                    vae_repo='stabilityai/sd-vae-ft-mse',
                    scheduler_repo='stable-diffusion-v1-5/stable-diffusion-inpainting')
        self.core = DeCoVTONPipeline.from_vton_checkpoint(local_files_only=True, **load).to(self.device)
        self.core.vae.enable_slicing()
        # Keep CUDA memory for garment detail. The safety checker still runs
        # on every returned image, in float32 on CPU.
        self.safety_device = torch.device('cpu')
        self.checker = _from_pretrained(
            StableDiffusionSafetyChecker.from_pretrained,
            'stable-diffusion-v1-5/stable-diffusion-inpainting', subfolder='safety_checker',
            torch_dtype=torch.float32, variant='fp16', use_safetensors=True).to(self.safety_device)
        self.extractor = _from_pretrained(
            CLIPImageProcessor.from_pretrained,
            'stable-diffusion-v1-5/stable-diffusion-inpainting', subfolder='feature_extractor')
        self.last_safety_blocked = False

    def tensor(self, image):
        device = getattr(self, 'device', torch.device('cpu'))
        rgb = np.asarray(image.convert('RGB')).copy()
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)/127.5-1

    @torch.inference_mode()
    def __call__(self, image, condition_image, mask, num_inference_steps=50,
                 guidance_scale=2.5, generator=None, **kwargs):
        self.last_safety_blocked = False
        device = getattr(self, 'device', torch.device('cpu'))
        size = self.resolution
        person = image.resize(size, Image.Resampling.LANCZOS)
        garment = ImageOps.pad(condition_image.convert('RGB'), size,
                               method=Image.Resampling.LANCZOS, color='white')
        mask_array = np.asarray(mask.convert('L').resize(size, Image.Resampling.BILINEAR)).copy()
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).float().to(device)/255
        mask_tensor = (mask_tensor >= .5).float()
        # DeCo-VTON expects a masked person; its pipeline does not mask pixels.
        condition = torch.cat((self.tensor(person)*(1-mask_tensor), self.tensor(garment)), dim=2)
        if generator is None:
            generator = torch.Generator(device=device).manual_seed(42)
        callback=kwargs.get('progress_callback')
        original_bar=getattr(self.core,'progress_bar',None)
        if callback is not None and original_bar is not None:
            def tracked_bar(iterable):
                for index,item in enumerate(original_bar(iterable)):
                    yield item
                    callback(index+1,len(iterable))
            self.core.progress_bar=tracked_bar
        try:
            results, _ = self.core(condition_image=condition, mask=mask_tensor,
                                   num_inference_steps=num_inference_steps,
                                   guidance_scale=guidance_scale, generator=generator,
                                   scheduler_type='ddim', eta=1.0)
        finally:
            if callback is not None and original_bar is not None:self.core.progress_bar=original_bar
        pixels = np.stack([np.asarray(frame.convert('RGB')) for frame in results])
        safety_device=getattr(self,'safety_device',device)
        safety_dtype=torch.float32 if safety_device.type=='cpu' else getattr(self,'dtype',torch.float16)
        features = self.extractor(results, return_tensors='pt').pixel_values.to(safety_device, dtype=safety_dtype)
        _, flags = self.checker(images=pixels, clip_input=features)
        self.last_safety_blocked = bool(any(flags))
        if self.last_safety_blocked:
            raise OutputBlocked('The generated image could not be returned. Try a different photo or garment.')
        return results
