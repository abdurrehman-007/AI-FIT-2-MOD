"""Local FASHN adapter with sequential guidance and CPU preprocessing/safety.

The upstream model/code is installed by setup_fashn.py. No user images leave
this process. Sequential CFG avoids doubling activations on a 6 GB GPU.
"""
import gc
import sys
import types
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from deco_backend import OutputBlocked, _from_pretrained


def report_step(model):
    callback=getattr(model,'_aifit_progress_callback',None)
    if callable(callback):
        callback(model._aifit_sample_step,model._aifit_sample_steps)


def sequential_guidance(model, *args, **kwargs):
    kwargs={k:v for k,v in kwargs.items() if v is not None}
    batch=args[0].shape[0]
    device=args[0].device
    conditional=model.forward(*args,**kwargs,mask=torch.ones(batch,device=device,dtype=torch.bool))['x']
    step=getattr(model,'_aifit_sample_step',None)
    if step is not None:
        model._aifit_sample_step=step+1
        if step>=model._aifit_sample_steps-model._aifit_skip_cfg:
            # Upstream sampling uses v_c alone here. Avoid computing a second
            # forward whose output the sampler will immediately discard.
            report_step(model)
            return {'v_c':conditional,'v_u':conditional}
    unconditional=model.forward(*args,**kwargs,mask=torch.zeros(batch,device=device,dtype=torch.bool))['x']
    if step is not None:report_step(model)
    return {'v_c':conditional,'v_u':unconditional}


class FashnBackend:
    resolution=(576,864)

    def resident_bytes(self):
        return sum(p.numel()*p.element_size() for p in self.core.tryon_model.parameters())

    def park_bytes(self):
        # Original BF16 tensors are file-backed, so parking allocates wrappers
        # rather than another dirty, approximately 1.9 GB host weight copy.
        return 16*1024**2

    def park_cpu(self):
        from safetensors.torch import load_file
        state=load_file(str(self.checkpoint/'model.safetensors'),device='cpu')
        self.core.tryon_model.load_state_dict(state,assign=True)

    def activate_cuda(self):
        self.core.tryon_model.to('cuda',non_blocking=True)
        torch.cuda.synchronize()

    def parse_preview(self, image):
        """Independent 18-class check for patterned long garments and trousers."""
        return self.core.hp_model.predict(image.convert('RGB'))

    def __init__(self, checkpoint):
        checkpoint=Path(checkpoint).resolve()
        self.checkpoint=checkpoint
        root=Path(__file__).resolve().parents[1]
        for path in (root/'.external/fashn-vton-1.5/src',root/'.runtime-deps/fashn'):
            sys.path.insert(0,str(path))
        try:
            from fashn_vton import TryOnPipeline
            from fashn_vton.tryon_mmdit import TryOnModel
            import fashn_vton.tryon_mmdit as model_module
            from fast_attention import attention
            model_module._attn_processor=attention
            from fashn_vton.dwpose import DWposeDetector
            from fashn_human_parser import FashnHumanParser
            from safetensors.torch import load_file
        except ImportError as error:
            raise RuntimeError('FASHN dependencies are missing. Run gpu_service/setup_fashn.py.') from error
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('The low-memory FASHN backend requires a CUDA GPU with bfloat16 support.')
        class LowMemoryPipeline(TryOnPipeline):
            def _sample(self,**kwargs):
                model=self.tryon_model
                model._aifit_sample_step=0
                model._aifit_sample_steps=kwargs.get('num_timesteps',30)
                model._aifit_skip_cfg=kwargs.get('skip_cfg_last_n_steps',1)
                try:
                    return super()._sample(**kwargs)
                finally:
                    model._aifit_sample_step=None

            def _setup_tryon_model(self):
                previous=torch.get_default_dtype()
                try:
                    torch.set_default_dtype(self.inference_dtype)
                    self.tryon_model=TryOnModel()
                finally:
                    torch.set_default_dtype(previous)
                state=load_file(str(checkpoint/'model.safetensors'),device='cpu')
                self.tryon_model.load_state_dict(state)
                del state
                gc.collect()
                self.tryon_model.to(self.device,dtype=self.inference_dtype).eval()
                self.tryon_model.forward_for_cfg=types.MethodType(sequential_guidance,self.tryon_model)

            def _setup_pose_model(self):
                self.pose_model=DWposeDetector(checkpoints_dir=str(checkpoint/'dwpose'),device='cpu')

            def _setup_hp_model(self):
                self.hp_model=FashnHumanParser(model_id=str(checkpoint/'parser'),device='cpu')

        self.core=LowMemoryPipeline(str(checkpoint),device='cuda')
        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
        from transformers import CLIPImageProcessor
        self.checker=_from_pretrained(StableDiffusionSafetyChecker.from_pretrained,
            'stable-diffusion-v1-5/stable-diffusion-inpainting',subfolder='safety_checker',
            torch_dtype=torch.float32,variant='fp16',use_safetensors=True).to('cpu')
        self.extractor=_from_pretrained(CLIPImageProcessor.from_pretrained,
            'stable-diffusion-v1-5/stable-diffusion-inpainting',subfolder='feature_extractor')
        self.last_safety_blocked=False

    @torch.inference_mode()
    def __call__(self,image,condition_image,mask=None,num_inference_steps=30,
                 guidance_scale=1.5,generator=None,category='upper',**kwargs):
        self.last_safety_blocked=False
        seed=generator.initial_seed() if generator is not None else 42
        category_name={'upper':'tops','lower':'bottoms','overall':'one-pieces'}[category]
        segmentation_free=kwargs.get('segmentation_free',True)
        model=getattr(self.core,'tryon_model',None)
        if model is not None:model._aifit_progress_callback=kwargs.get('progress_callback')
        try:
            result=self.core(image.convert('RGB'),condition_image.convert('RGB'),category=category_name,
                garment_photo_type='flat-lay',num_timesteps=num_inference_steps,
                guidance_scale=guidance_scale,seed=seed,segmentation_free=segmentation_free,
                skip_cfg_last_n_steps=3 if num_inference_steps<=20 else 1)
        finally:
            if model is not None:model._aifit_progress_callback=None
        frames=[frame.convert('RGB').resize(image.size,Image.Resampling.LANCZOS) for frame in result.images]
        features=self.extractor(frames,return_tensors='pt').pixel_values.to('cpu',dtype=torch.float32)
        _,flags=self.checker(images=np.stack([np.asarray(frame) for frame in frames]),clip_input=features)
        self.last_safety_blocked=bool(any(flags))
        if self.last_safety_blocked:
            raise OutputBlocked('The generated image could not be returned. Try a different photo or garment.')
        return frames
