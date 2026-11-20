import unittest
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from deco_backend import DeCoBackend, OutputBlocked


class BackendTests(unittest.TestCase):
    def test_progress_tracks_completed_steps_and_restores_pipeline_bar(self):
        events=[]
        class Core:
            def progress_bar(self,items):return iter(items)
            def __call__(self,**kwargs):
                for step in self.progress_bar(range(3)):events.append(('work',step))
                return [Image.new('RGB',(384,512),'blue')],[]
        self.backend.core=Core();original=self.backend.core.progress_bar
        self.backend(image=self.person,condition_image=self.garment,mask=self.mask,
            progress_callback=lambda done,total:events.append(('progress',done,total)))
        self.assertEqual(events,[('work',0),('progress',1,3),('work',1),('progress',2,3),('work',2),('progress',3,3)])
        self.assertEqual(self.backend.core.progress_bar,original)

    def setUp(self):
        self.backend = DeCoBackend.__new__(DeCoBackend)
        self.backend.dtype = torch.float16
        self.backend.extractor = lambda *a, **kw: SimpleNamespace(pixel_values=SimpleNamespace(to=lambda *a, **kw: None))
        self.flags = [False]
        self.backend.checker = lambda **kw: (None, self.flags)
        self.calls = []
        def core(**kwargs):
            self.calls.append(kwargs)
            return [Image.new('RGB', (384, 512), 'blue')], []
        self.backend.core = core
        self.person = Image.new('RGB', (384, 512), 'red')
        self.garment = Image.new('RGB', (384, 512), 'blue')
        array = np.zeros((512, 384), dtype=np.uint8)
        array[100:300, 100:300] = 255
        self.mask = Image.fromarray(array)

    def run_backend(self):
        return self.backend(image=self.person, condition_image=self.garment, mask=self.mask)

    def test_condition_contains_masked_person_and_rgb_garment(self):
        self.run_backend()
        condition = self.calls[0]['condition_image']
        self.assertEqual(tuple(condition.shape), (1, 3, 1024, 384))
        self.assertEqual(condition[0, :, 200, 200].tolist(), [0, 0, 0])
        self.assertEqual(condition[0, :, 20, 20].tolist(), [1, -1, -1])
        self.assertEqual(condition[0, :, 700, 200].tolist(), [-1, -1, 1])

    def test_blocked_output_is_never_returned(self):
        self.flags[0] = True
        with self.assertRaises(OutputBlocked): self.run_backend()
        self.assertTrue(self.backend.last_safety_blocked)

    def test_blocked_state_is_reset_after_subsequent_safe_result(self):
        self.backend.last_safety_blocked = True
        self.assertEqual(len(self.run_backend()), 1)
        self.assertFalse(self.backend.last_safety_blocked)


if __name__ == '__main__': unittest.main()
