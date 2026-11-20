"""Audit persistence checks without GPU inference or real inventory writes."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import urllib.error
import audit_upper_tryon as audit


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root/'instance').mkdir()
        (self.root/'instance/gpu.key').write_text('test-key')
        self.folder = self.root/'audit';self.folder.mkdir()
        (self.folder/'outputs').mkdir()
        self.service_check=audit.ensure_service
        self.patches = [patch.object(audit,'ROOT',self.root), patch.object(audit,'FOLDER',self.folder),
                        patch.object(audit,'ensure_service'),
                        patch.object(audit,'source_version',return_value='v1')]
        for item in self.patches:
            item.start();self.addCleanup(item.stop)

    def test_diagonal_plan_covers_every_pair_and_styles_in_first_round(self):
        photos=list(range(7));products=list(range(3))
        pairs=list(audit.ordered_pairs(photos,products))
        self.assertEqual(len(set(pairs)),21)
        self.assertEqual({photo for photo,product in pairs[:7]},set(photos))
        self.assertEqual({product for photo,product in pairs[:7]},set(products))

    def test_different_loaded_service_is_refused_before_inference(self):
        opener=Mock();opener.open.return_value=io.BytesIO(b'{"ready":true,"source_fingerprint":"old"}')
        with self.assertRaisesRegex(RuntimeError,'fingerprint differs'):
            self.service_check(opener,'new')
        opener.open.return_value=io.BytesIO(b'{"ready":true,"source_fingerprint":"new"}')
        self.service_check(opener,'new')

    def test_upper_length_is_sent_and_is_part_of_pair_identity(self):
        m=self.manifest('ready');photo=m['photos'][0];product=m['products'][0]
        standard=audit.pair_id(photo,product,'v1')
        product['upper_length']='long'
        self.assertNotEqual(standard,audit.pair_id(photo,product,'v1'))
        payload,_=audit.encode_request(photo,product)
        self.assertIn(b'name="upper_length"\r\n\r\nlong\r\n',payload)

    def manifest(self, status='input_rejected'):
        photo = self.root/'source.png';photo.write_bytes(b'source')
        product = self.root/'garment.png';product.write_bytes(b'garment')
        manifest={'complete':True,'source_version':'v1','total_pairs':1,
                  'photos':[{'name':'person.png','status':status,'error':'Two people',
                             'sha256':audit.digest(photo),'path':str(photo)}],
                  'products':[{'id':'product1','name':'Shirt','status':'ready',
                               'sha256':audit.digest(product),'path':str(product)}]}
        audit.save_json(self.folder/'manifest.json',manifest)
        return manifest

    def test_rejected_input_is_recorded_and_resume_does_not_duplicate(self):
        self.manifest()
        with patch.object(audit.urllib.request,'build_opener') as opener:
            audit.run();audit.run()
        opener.return_value.open.assert_not_called()
        rows=(self.folder/'results.jsonl').read_text().splitlines()
        self.assertEqual(len(rows),1)
        self.assertEqual(json.loads(rows[0])['status'],'input_rejected')
        progress=json.loads((self.folder/'progress.json').read_text())
        self.assertEqual(progress['finished_pairs'],1)
        self.assertEqual(progress['remaining_pairs'],0)

    def test_service_error_does_not_mark_case_complete(self):
        self.manifest('ready')
        error=urllib.error.HTTPError('http://local/infer',500,'Error',{},io.BytesIO(b'{"error":"failed inference"}'))
        opener=Mock();opener.open.side_effect=error
        with patch.object(audit.urllib.request,'build_opener',return_value=opener):
            with self.assertRaisesRegex(RuntimeError,'failed inference'):
                audit.run()
        self.assertFalse((self.folder/'results.jsonl').exists())

    def test_stop_is_honoured_while_yielding_to_an_interactive_request(self):
        self.manifest('ready')
        error=urllib.error.HTTPError('http://local/infer',429,'Busy',{},io.BytesIO(b'{}'))
        opener=Mock();opener.open.side_effect=error
        with patch.object(audit.urllib.request,'build_opener',return_value=opener),patch.object(audit.time,'sleep',side_effect=lambda seconds:(self.folder/'STOP').write_text('stop')):
            audit.run()
        self.assertEqual(opener.open.call_count,1)
        self.assertFalse((self.folder/'results.jsonl').exists())

    def test_changed_code_or_input_cannot_reuse_old_results(self):
        manifest=self.manifest('ready')
        with patch.object(audit,'source_version',return_value='v2'):
            with self.assertRaisesRegex(RuntimeError,'source changed'):
                audit.run()
        Path(manifest['photos'][0]['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'input file changed'):
            audit.run()

    def test_incomplete_final_journal_line_is_repaired(self):
        journal=self.folder/'results.jsonl'
        journal.write_bytes(b'{"case":"valid"}\n{"case":"unfinished')
        self.assertEqual(list(audit.load_results(journal)),['valid'])
        self.assertEqual(journal.read_bytes(),b'{"case":"valid"}\n')
        journal.write_bytes(b'{"case":"valid"}')
        audit.load_results(journal)
        self.assertTrue(journal.read_bytes().endswith(b'\n'))

    def test_interior_journal_damage_is_not_silently_dropped(self):
        journal=self.folder/'results.jsonl'
        original=b'corrupt\n{"case":"valid"}\n'
        journal.write_bytes(original)
        with self.assertRaisesRegex(RuntimeError,'interior damage'):
            audit.load_results(journal)
        self.assertEqual(journal.read_bytes(),original)


if __name__ == '__main__':
    unittest.main()
