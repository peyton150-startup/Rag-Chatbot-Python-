import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


def load_embeddings_class():
    try:
        module = importlib.import_module("nvidia_embeddings")
    except ModuleNotFoundError:
        return None
    return module.NVIDIAOpenAIEmbeddings


class FakeEmbeddingsEndpoint:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        data = [
            SimpleNamespace(
                object="embedding",
                index=index,
                embedding=[float(len(text)), float(index)],
            )
            for index, text in enumerate(kwargs["input"])
        ]
        return SimpleNamespace(
            object="list",
            model=kwargs["model"],
            data=data,
            usage=SimpleNamespace(prompt_tokens=0, total_tokens=0),
        )


class FakeOpenAIClient:
    def __init__(self):
        self.embeddings = FakeEmbeddingsEndpoint()


class NVIDIAOpenAIEmbeddingsTests(unittest.TestCase):
    def setUp(self):
        self.embeddings_class = load_embeddings_class()
        self.assertIsNotNone(
            self.embeddings_class,
            "nvidia_embeddings.NVIDIAOpenAIEmbeddings must exist",
        )

    def test_documents_use_passage_mode(self):
        client = FakeOpenAIClient()
        embeddings = self.embeddings_class(client=client)

        vectors = embeddings.embed_documents(["alpha", "bravo"])

        self.assertEqual(vectors, [[5.0, 0.0], [5.0, 1.0]])
        self.assertEqual(client.embeddings.calls[0]["extra_body"]["input_type"], "passage")

    def test_query_uses_query_mode(self):
        client = FakeOpenAIClient()
        embeddings = self.embeddings_class(client=client)

        vector = embeddings.embed_query("question")

        self.assertEqual(vector, [8.0, 0.0])
        self.assertEqual(client.embeddings.calls[0]["extra_body"]["input_type"], "query")

    def test_documents_are_batched_without_losing_order(self):
        client = FakeOpenAIClient()
        embeddings = self.embeddings_class(client=client, batch_size=2)

        vectors = embeddings.embed_documents(["a", "bb", "ccc"])

        self.assertEqual(vectors, [[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
        self.assertEqual(len(client.embeddings.calls), 2)

    def test_missing_api_key_is_rejected_before_client_creation(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "NVIDIA_API_KEY"):
                self.embeddings_class()


if __name__ == "__main__":
    unittest.main()
