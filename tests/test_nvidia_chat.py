import importlib
import os
import unittest
from unittest.mock import patch


def load_chat_factory():
    try:
        module = importlib.import_module("nvidia_chat")
    except ModuleNotFoundError:
        return None
    return module.create_nvidia_chat_model


class NVIDIAChatTests(unittest.TestCase):
    def setUp(self):
        self.create_chat_model = load_chat_factory()
        self.assertIsNotNone(
            self.create_chat_model,
            "nvidia_chat.create_nvidia_chat_model must exist",
        )

    def test_chat_model_targets_lightning_on_nvidia_api(self):
        model = self.create_chat_model(api_key="test-key")

        self.assertEqual(model.model_name, "nvidia/nemotron-3.5-lightning-30b-a3b")
        self.assertEqual(model.openai_api_base, "https://integrate.api.nvidia.com/v1")
        self.assertFalse(model.streaming)

    def test_thinking_mode_is_turned_off(self):
        model = self.create_chat_model(api_key="test-key")

        self.assertEqual(
            model._default_params["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_missing_api_key_is_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "NVIDIA_API_KEY"):
                self.create_chat_model()


if __name__ == "__main__":
    unittest.main()
