import contextlib
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import download_deco


class DownloadTests(unittest.TestCase):
    def test_interrupted_download_resumes_and_checks_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/'.models/deco/unet'
            folder.mkdir(parents=True)
            (folder/'weights.partial').write_bytes(b'abc')
            response = SimpleNamespace(status_code=206, headers={'Content-Range': 'bytes 3-5/6'},
                                       raise_for_status=lambda: None, iter_content=lambda size: [b'def'])
            files = {'weights': (6, hashlib.sha256(b'abcdef').hexdigest())}
            with patch.object(download_deco, 'ROOT', root), patch.object(download_deco, 'FILES', files), \
                 patch.object(download_deco.requests, 'get', return_value=contextlib.nullcontext(response)) as get:
                download_deco.download()
            self.assertEqual(get.call_args.kwargs['headers'], {'Range': 'bytes=3-'})
            self.assertEqual((folder/'weights').read_bytes(), b'abcdef')
            self.assertFalse((folder/'weights.partial').exists())

    def test_server_ignoring_range_restarts_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/'.models/deco/unet'
            folder.mkdir(parents=True)
            (folder/'weights.partial').write_bytes(b'abc')
            response = SimpleNamespace(status_code=200, headers={}, raise_for_status=lambda: None,
                                       iter_content=lambda size: [b'abcdef'])
            files = {'weights': (6, hashlib.sha256(b'abcdef').hexdigest())}
            with patch.object(download_deco, 'ROOT', root), patch.object(download_deco, 'FILES', files), \
                 patch.object(download_deco.requests, 'get', return_value=contextlib.nullcontext(response)):
                download_deco.download()
            self.assertEqual((folder/'weights').read_bytes(), b'abcdef')

    def test_wrong_contents_are_not_installed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            response = SimpleNamespace(status_code=200, headers={}, raise_for_status=lambda: None,
                                       iter_content=lambda size: [b'broken'])
            files = {'weights': (6, hashlib.sha256(b'abcdef').hexdigest())}
            with patch.object(download_deco, 'ROOT', root), patch.object(download_deco, 'FILES', files), \
                 patch.object(download_deco.requests, 'get', return_value=contextlib.nullcontext(response)):
                with self.assertRaises(RuntimeError): download_deco.download()
            self.assertFalse((root/'.models/deco/unet/weights').exists())


if __name__ == '__main__': unittest.main()
