-- AI-FIT SQLite schema

CREATE TABLE documents (collection TEXT, id TEXT, body TEXT NOT NULL, PRIMARY KEY(collection,id));

