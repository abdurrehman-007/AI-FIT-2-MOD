"""Bounded, thread-safe inference progress; percentages and ETA are estimates."""
import threading
import time
from collections import OrderedDict


class Progress:
    def __init__(self, clock=time.monotonic, capacity=128, ttl=1200):
        self.clock=clock;self.capacity=capacity;self.ttl=ttl
        self.rows=OrderedDict();self.lock=threading.Lock()

    def start(self, key):
        if not key:return
        with self.lock:
            now=self.clock()
            for old,row in list(self.rows.items()):
                if now-row['updated']>self.ttl:del self.rows[old]
            if key in self.rows:raise ValueError('Duplicate progress identifier')
            # Never evict a running request just to admit another tracker.
            if len(self.rows)>=self.capacity:
                finished=next((k for k,r in self.rows.items() if r['status']!='running'),None)
                if finished is None:raise ValueError('Too many tracked requests')
                del self.rows[finished]
            self.rows[key]=dict(started=now,updated=now,stage='queued',percent=0,
                status='running',eta=None,estimated=True)

    def update(self, key, stage, percent, remaining=None, **extra):
        if not key:return
        with self.lock:
            row=self.rows.get(key)
            if row is None or row['status']!='running':return
            now=self.clock()
            row.update(stage=stage,percent=max(row['percent'],min(99,percent)),updated=now,**extra)
            row['eta']=now+max(0,remaining) if remaining is not None else None

    def finish(self, key, ok):
        if not key:return
        with self.lock:
            row=self.rows.get(key)
            if row is not None:
                row.update(status='complete' if ok else 'failed',stage='complete' if ok else 'failed',
                    percent=100 if ok else row['percent'],updated=self.clock(),eta=None)

    def snapshot(self, key):
        with self.lock:
            row=self.rows.get(key)
            if row is None or self.clock()-row['updated']>self.ttl:return None
            now=self.clock();elapsed=(now if row['status']=='running' else row['updated'])-row['started']
            remaining=None if row['eta'] is None else max(0,row['eta']-now)
            return dict(stage=row['stage'],status=row['status'],percent=round(row['percent']),
                elapsed_seconds=round(elapsed,1),remaining_seconds=None if remaining is None else round(remaining,1),
                estimated_total_seconds=None if remaining is None else round(elapsed+remaining,1),
                estimated=True,attempt=row.get('attempt',1))

    def steps(self, key, total, start_percent, end_percent, tail=7, synchronize=None, initial_rate=None, stage='generating'):
        started=self.clock();last=[started-2];anchor=[None,None]
        def notify(done, count):
            now=self.clock()
            if done<count and now-last[0]<1:return
            if synchronize is not None:synchronize()
            now=self.clock();last[0]=now
            if anchor[0] is None:
                rate=initial_rate if initial_rate is not None else (now-started)/max(1,done)
                anchor[:]=[done,now]
            else:rate=(now-anchor[1])/max(1,done-anchor[0])
            self.update(key,stage,start_percent+(end_percent-start_percent)*done/max(1,count),
                remaining=max(0,count-done)*rate+tail)
        return notify if key else None
