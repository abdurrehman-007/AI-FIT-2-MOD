import unittest
from types import SimpleNamespace
from PIL import Image
import torch
from fashn_backend import FashnBackend,sequential_guidance
from deco_backend import OutputBlocked


class FashnBackendTests(unittest.TestCase):
    def test_failed_inference_drops_request_progress_callback(self):
        class Core:
            tryon_model=SimpleNamespace()
            def __call__(self,*args,**kwargs):raise RuntimeError('simulated inference failure')
        backend=FashnBackend.__new__(FashnBackend);backend.core=Core()
        with self.assertRaisesRegex(RuntimeError,'simulated'):
            backend(image=Image.new('RGB',(80,100)),condition_image=Image.new('RGB',(40,60)),progress_callback=lambda *a:None)
        self.assertIsNone(backend.core.tryon_model._aifit_progress_callback)

    def test_progress_is_reported_after_actual_conditional_and_unconditional_work(self):
        events=[]
        class Model:
            _aifit_sample_step=0;_aifit_sample_steps=3;_aifit_skip_cfg=1
            def _aifit_progress_callback(self,done,total):events.append(('progress',done,total))
            def forward(self,x,t,mask,**kwargs):events.append(('forward',bool(mask[0])));return {'x':x}
        model=Model();x=torch.ones(1,3,4,4)
        for _ in range(3):sequential_guidance(model,x,torch.zeros(1))
        self.assertEqual(events,[('forward',True),('forward',False),('progress',1,3),
            ('forward',True),('forward',False),('progress',2,3),('forward',True),('progress',3,3)])

    def test_file_backed_parking_preserves_exact_weights_and_eval_state(self):
        import tempfile
        from pathlib import Path
        from safetensors.torch import save_file
        model=torch.nn.Linear(4,4).to(dtype=torch.bfloat16).eval()
        expected={k:v.detach().clone() for k,v in model.state_dict().items()}
        backend=FashnBackend.__new__(FashnBackend)
        backend.core=SimpleNamespace(tryon_model=model)
        with tempfile.TemporaryDirectory() as directory:
            backend.checkpoint=Path(directory)
            save_file(expected,str(backend.checkpoint/'model.safetensors'))
            with torch.no_grad():model.weight.zero_()
            backend.park_cpu()
            for k,v in model.state_dict().items():
                torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
            self.assertFalse(model.training)
            self.assertEqual(backend.park_bytes(),16*1024**2)
            # Release mapped weights before TemporaryDirectory removes them on Windows.
            model.to(dtype=torch.float32)

    def test_sampler_final_steps_do_not_compute_discarded_unconditional_pass(self):
        calls=[]
        class Model:
            _aifit_sample_step=0
            _aifit_sample_steps=5
            _aifit_skip_cfg=2
            def forward(self,x,t,mask,ca_images):
                calls.append(bool(mask[0]))
                return {'x':x+ca_images*mask[:,None,None,None]}
        model=Model();x=torch.ones(1,3,4,4)
        for step in range(5):
            result=sequential_guidance(model,x,torch.zeros(1),ca_images=x)
            torch.testing.assert_close(result['v_c'],x*2)
            if step<3:torch.testing.assert_close(result['v_u'],x)
        self.assertEqual(calls,[True,False]*3+[True,True])

    def test_sequential_guidance_matches_batched_condition_order(self):
        batches=[]
        class Model:
            def forward(self,x,t,mask,ca_images):
                batches.append(x.shape[0])
                return {'x':x+ca_images*mask[:,None,None,None]}
        model=Model();image=torch.randn(2,3,4,4);condition=torch.randn_like(image)
        expected=model.forward(torch.cat((image,image)),torch.zeros(4),
            mask=torch.tensor([True,True,False,False]),ca_images=torch.cat((condition,condition)))['x']
        batches.clear()
        result=sequential_guidance(model,image,torch.zeros(2),ca_images=condition,unused=None)
        torch.testing.assert_close(result['v_c'],expected[:2])
        torch.testing.assert_close(result['v_u'],expected[2:])
        self.assertEqual(batches,[2,2])

    def backend(self):
        backend=FashnBackend.__new__(FashnBackend)
        calls=[]
        def core(*args,**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[Image.new('RGB',(40,60),'blue')])
        backend.core=core
        backend.extractor=lambda *a,**kw:SimpleNamespace(pixel_values=torch.zeros(1,3,20,20))
        backend.flags=[False]
        backend.checker=lambda **kw:(None,backend.flags)
        return backend,calls

    def test_category_seed_and_source_dimensions_are_preserved(self):
        backend,calls=self.backend()
        source=Image.new('RGB',(80,100));garment=Image.new('RGB',(40,60))
        images=backend(image=source,condition_image=garment,category='lower',
            generator=torch.Generator().manual_seed(73))
        self.assertEqual(images[0].size,source.size)
        self.assertEqual(calls[0]['category'],'bottoms')
        self.assertEqual(calls[0]['seed'],73)
        self.assertTrue(calls[0]['segmentation_free'])

    def test_safety_block_cannot_return_an_image_and_state_resets(self):
        backend,_=self.backend();backend.flags=[True]
        args={'image':Image.new('RGB',(80,100)),'condition_image':Image.new('RGB',(40,60))}
        with self.assertRaises(OutputBlocked):backend(**args)
        self.assertTrue(backend.last_safety_blocked)
        backend.flags=[False]
        self.assertEqual(len(backend(**args)),1)
        self.assertFalse(backend.last_safety_blocked)
