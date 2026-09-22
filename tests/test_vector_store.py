import importlib
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from contextlib import closing
from unittest.mock import patch

import chromadb
import streamlit as st
from chromadb.api.shared_system_client import SharedSystemClient


def load_vector_store_loader():
    try:
        module = importlib.import_module("vector_store")
    except ModuleNotFoundError:
        return None
    return module.load_vector_store


def close_chroma_clients():
    for system in SharedSystemClient._identifier_to_system.values():
        system.stop()
    SharedSystemClient.clear_system_cache()


def make_chromadb_0_4_store(path):
    """Build a store with the schema ingest.py wrote under chromadb 0.4.24,
    before chromadb 0.5 added collections.config_json_str."""
    settings = chromadb.config.Settings(anonymized_telemetry=False)
    chromadb.PersistentClient(path=path, settings=settings).get_or_create_collection("langchain")
    close_chroma_clients()
    with closing(sqlite3.connect(os.path.join(path, "chroma.sqlite3"))) as con:
        con.execute("ALTER TABLE collections DROP COLUMN config_json_str")
        con.execute("DELETE FROM migrations WHERE dir = 'sysdb' AND version = 7")
        con.commit()


def open_from_concurrent_sessions(open_store, path, sessions=4):
    barrier = threading.Barrier(sessions)
    outcomes = [None] * sessions

    def session(index):
        barrier.wait()
        try:
            outcomes[index] = open_store(path)
        except Exception as error:
            outcomes[index] = error

    threads = [threading.Thread(target=session, args=(i,)) for i in range(sessions)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return outcomes


class LoadVectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.load_vector_store = load_vector_store_loader()
        self.assertIsNotNone(
            self.load_vector_store,
            "vector_store.load_vector_store must exist",
        )
        env = patch.dict(
            os.environ,
            {"NVIDIA_API_KEY": "test-key", "ANONYMIZED_TELEMETRY": "False"},
        )
        env.start()
        self.addCleanup(env.stop)

        temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(temp_dir.cleanup)
        self.addCleanup(close_chroma_clients)
        self.addCleanup(st.cache_resource.clear)
        self.path = temp_dir.name

    def test_concurrent_sessions_can_open_a_store_written_by_chromadb_0_4(self):
        make_chromadb_0_4_store(self.path)

        outcomes = open_from_concurrent_sessions(self.load_vector_store, self.path)

        errors = [repr(outcome) for outcome in outcomes if isinstance(outcome, Exception)]
        self.assertEqual(errors, [])

    def test_concurrent_sessions_open_the_store_one_at_a_time(self):
        lock = threading.Lock()
        opening = 0
        most_at_once = 0

        def slow_open(**kwargs):
            nonlocal opening, most_at_once
            with lock:
                opening += 1
                most_at_once = max(most_at_once, opening)
            time.sleep(0.2)
            with lock:
                opening -= 1
            return object()

        with patch("vector_store.Chroma", side_effect=slow_open):
            outcomes = open_from_concurrent_sessions(self.load_vector_store, self.path)

        self.assertEqual(most_at_once, 1)
        self.assertFalse(any(isinstance(outcome, Exception) for outcome in outcomes))


if __name__ == "__main__":
    unittest.main()
