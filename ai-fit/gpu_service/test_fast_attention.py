import unittest
from contextlib import nullcontext
from unittest.mock import patch
import torch
import fast_attention as module


class AttentionTests(unittest.TestCase):
    def test_cpu_path_preserves_original_math(self):
        q,k,v=[torch.randn(1,2,8,4) for _ in range(3)]
        actual=module.NativeAttention()(q,k,v)
        torch.testing.assert_close(actual,torch.nn.functional.scaled_dot_product_attention(q,k,v),rtol=0,atol=0)

    def test_unsupported_kernel_falls_back_without_repeating_failed_path(self):
        q=torch.ones(1,2,8,4);attention=module.NativeAttention()
        with patch.object(module,'candidate',return_value=True),patch.object(module,'eligible',return_value=True),patch.object(module,'sdpa_kernel',side_effect=lambda *a:nullcontext()),patch.object(module,'scaled_dot_product_attention',side_effect=[RuntimeError('No available kernel'),q,q]) as call,patch.object(module.torch.cuda,'empty_cache'):
            self.assertIs(attention(q,q,q),q)
            self.assertIs(attention(q,q,q),q)
            self.assertEqual(call.call_count,3)
            self.assertEqual(attention.original_calls,2)
            self.assertEqual(len(attention.disabled),1)

    def test_unexpected_runtime_error_is_not_hidden(self):
        q=torch.ones(1,2,8,4)
        with patch.object(module,'candidate',return_value=True),patch.object(module,'eligible',return_value=True),patch.object(module,'sdpa_kernel',side_effect=lambda *a:nullcontext()),patch.object(module,'scaled_dot_product_attention',side_effect=RuntimeError('device assertion')):
            with self.assertRaisesRegex(RuntimeError,'device assertion'):module.NativeAttention()(q,q,q)
