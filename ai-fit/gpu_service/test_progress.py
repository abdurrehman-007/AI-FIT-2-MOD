import unittest
from progress import Progress


class ProgressTests(unittest.TestCase):
    def test_preprocessing_time_does_not_inflate_sampling_rate(self):
        now=[0.];p=Progress(clock=lambda:now[0]);p.start('a')
        callback=p.steps('a',20,20,80,initial_rate=1)
        now[0]=10;callback(1,20)
        self.assertEqual(p.snapshot('a')['remaining_seconds'],26)
        now[0]=12;callback(3,20)
        self.assertEqual(p.snapshot('a')['remaining_seconds'],24)

    def test_step_eta_monotonic_percent_and_completion_only_on_success(self):
        now=[0.];p=Progress(clock=lambda:now[0]);p.start('a')
        callback=p.steps('a',20,20,80)
        now[0]=10;callback(5,20)
        snapshot=p.snapshot('a')
        self.assertEqual(snapshot['percent'],35)
        self.assertEqual(snapshot['remaining_seconds'],37)
        p.update('a','refining',25,remaining=50,attempt=2)
        self.assertEqual(p.snapshot('a')['percent'],35)
        p.update('a','saving',100,remaining=0)
        self.assertEqual(p.snapshot('a')['percent'],99)
        p.finish('a',False)
        self.assertEqual(p.snapshot('a')['status'],'failed')
        self.assertNotEqual(p.snapshot('a')['percent'],100)
        p.start('b');p.finish('b',True)
        self.assertEqual(p.snapshot('b')['percent'],100)

    def test_capacity_preserves_active_jobs_and_expiry_clears_state(self):
        now=[0.];p=Progress(clock=lambda:now[0],capacity=1,ttl=10);p.start('a')
        with self.assertRaises(ValueError):p.start('b')
        p.finish('a',True);p.start('b')
        self.assertIsNone(p.snapshot('a'))
        now[0]=11;self.assertIsNone(p.snapshot('b'))
        p.start('c');self.assertEqual(len(p.rows),1)
