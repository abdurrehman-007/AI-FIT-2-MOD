import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from runtime_fingerprint import code_fingerprint,runtime_fingerprint


class RuntimeFingerprintTests(unittest.TestCase):
    def test_loaded_code_and_calibration_changes_cannot_reuse_current_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'gpu_service').mkdir();(root/'instance').mkdir()
            code=root/'gpu_service/backend.py';code.write_text('version=1')
            loaded=code_fingerprint(root);before=runtime_fingerprint(root,loaded)
            code.write_text('version=2')
            self.assertEqual(before,runtime_fingerprint(root,loaded))
            self.assertNotEqual(before,runtime_fingerprint(root))
            (root/'instance/garment-backends.json').write_text('{"garments":{}}')
            self.assertNotEqual(before,runtime_fingerprint(root,loaded))
            current=runtime_fingerprint(root)
            with patch.dict('os.environ',{'AIFIT_ENGINE':'catvton'}):
                self.assertNotEqual(current,runtime_fingerprint(root))
            with patch.dict('os.environ',{'AIFIT_SPEED_PROFILE':'fast'}):
                fast=runtime_fingerprint(root)
                with patch.dict('os.environ',{'AIFIT_SPEED_PROFILE':'balanced'}):
                    self.assertNotEqual(fast,runtime_fingerprint(root))
            with patch.dict('os.environ',{'AIFIT_FASHN_ATTENTION':'original'}):
                original=runtime_fingerprint(root)
                with patch.dict('os.environ',{'AIFIT_FASHN_ATTENTION':'auto'}):
                    self.assertNotEqual(original,runtime_fingerprint(root))


if __name__=='__main__':unittest.main()
