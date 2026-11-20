"""Document persistence: SQLite locally, Firestore in configured deployments."""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class Store:
    def __init__(self, directory, backend='sqlite'):
        self.backend = backend
        if backend == 'firestore':
            import firebase_admin
            from firebase_admin import firestore
            if not firebase_admin._apps:
                firebase_admin.initialize_app()
            self.remote = firestore.client()
        else:
            self.path = Path(directory) / 'aifit.sqlite3'
            with self.connect() as db:
                db.execute('CREATE TABLE IF NOT EXISTS documents (collection TEXT, id TEXT, body TEXT NOT NULL, PRIMARY KEY(collection,id))')

    @contextmanager
    def connect(self):
        connection=sqlite3.connect(self.path, timeout=20)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def get(self, collection, key):
        if self.backend == 'firestore':
            return self.remote.collection(collection).document(key).get().to_dict()
        with self.connect() as db:
            row = db.execute('SELECT body FROM documents WHERE collection=? AND id=?', (collection,key)).fetchone()
        return json.loads(row[0]) if row else None

    def all(self, collection):
        if self.backend == 'firestore':
            return [x.to_dict() for x in self.remote.collection(collection).stream()]
        with self.connect() as db:
            return [json.loads(x[0]) for x in db.execute('SELECT body FROM documents WHERE collection=?', (collection,))]

    def put(self, collection, key, data):
        self.put_many(collection, [(key, data)])

    def put_many(self, collection, pairs):
        if self.backend == 'firestore':
            batch = self.remote.batch()
            for key,data in pairs:
                batch.set(self.remote.collection(collection).document(key), data)
            batch.commit()
        else:
            with self.connect() as db:
                db.executemany('INSERT INTO documents VALUES(?,?,?) ON CONFLICT(collection,id) DO UPDATE SET body=excluded.body', [(collection,key,json.dumps(data)) for key,data in pairs])

    def create(self, collection, key, data):
        if self.backend == 'firestore':
            from google.api_core.exceptions import AlreadyExists
            try:
                self.remote.collection(collection).document(key).create(data)
                return True
            except AlreadyExists:
                return False
        with self.connect() as db:
            return db.execute('INSERT OR IGNORE INTO documents VALUES(?,?,?)', (collection,key,json.dumps(data))).rowcount == 1

    def delete(self, collection, key):
        if self.backend == 'firestore':
            self.remote.collection(collection).document(key).delete()
        else:
            with self.connect() as db:
                db.execute('DELETE FROM documents WHERE collection=? AND id=?', (collection,key))
