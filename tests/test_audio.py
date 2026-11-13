"""Offline regression checks for script handling, timing, and model-download safety."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from img2vid import audio


class AudioTests(unittest.TestCase):
    def test_markdown_reads_only_fenced_narration(self):
        script = "# Title\nIgnore this style.\n```text\n[Spoken Intro]\nHello!\n[Short Instrumental Break]\nG is for giraffe.\n```\nMore instructions."
        items = audio.parse_script(script)
        self.assertEqual([s.get("text") for s in items], ["Hello!", None, "G is for giraffe."])
        self.assertEqual(items[1]["seconds"], 2)
        self.assertTrue(all(s["section"] == "Spoken Intro" for s in items))

    def test_explicit_pause_overrides_default(self):
        items = audio.parse_script("Hello.\n[Pause 3 seconds]\nAgain.", default_pause=1)
        self.assertEqual(items[1]["seconds"], 3)

    def test_unknown_directions_are_not_spoken(self):
        with self.assertRaisesRegex(ValueError, "unknown stage"):
            audio.parse_script("[Dance around]\nHello.")

    def test_invalid_inputs_fail(self):
        for content, structured in [
            ("", False), ("[Pause 3 seconds]", False), ('{"text":"A"}', True),
            ('[{"text":"A","pause":1}]', True), ('[{"text":false}]', True),
            ('[{"pause":-1},{"text":"A"}]', True), ('[{"pause":NaN},{"text":"A"}]', True),
        ]:
            with self.subTest(content=content), self.assertRaises(ValueError):
                audio.parse_script(content, is_json=structured)

    def test_letter_names_preserve_lowercase_articles(self):
        text = audio.pronounce_letters("G, H, I. Capital P and lowercase p. Give a clap.")
        self.assertEqual(text, "gee, aitch, eye. Capital pee and lowercase pee. Give a clap.")
        self.assertEqual(audio.pronounce_letters("Z.", british=True), "zed.")

    def test_pause_samples_timestamps_and_cache(self):
        engine = Mock()
        engine.create.return_value = (np.full(audio.RATE, 0.2, dtype=np.float32), audio.RATE)
        items = audio.parse_script("Hello.\n[pause 2.5]\nAgain.")
        with tempfile.TemporaryDirectory() as root, patch.object(audio, "cache_dir", return_value=Path(root)):
            samples, timeline = audio.synthesize(items, engine, "af_heart", 0.9)
            self.assertEqual(len(samples), int(4.5 * audio.RATE))
            self.assertTrue(np.all(samples[audio.RATE:int(3.5 * audio.RATE)] == 0))
            self.assertEqual(timeline[2]["start"], 3.5)
            self.assertEqual(timeline[2]["end"], 4.5)
            audio.synthesize(items, engine, "af_heart", 0.9)
            self.assertEqual(engine.create.call_count, 2)

    def test_missing_models_never_download_implicitly(self):
        with tempfile.TemporaryDirectory() as root, patch.object(audio, "cache_dir", return_value=Path(root)):
            with patch.object(audio.urllib.request, "urlopen") as network:
                with self.assertRaisesRegex(RuntimeError, "download-models"):
                    audio.model_files()
                network.assert_not_called()

    def test_bad_download_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as root, patch.object(audio, "cache_dir", return_value=Path(root)):
            existing = Path(root) / "model"
            existing.write_bytes(b"old!")
            checksum = hashlib.sha256(b"good").hexdigest()
            with patch.object(audio, "FILES", {"model": (4, checksum)}):
                with patch.object(audio.urllib.request, "urlopen", return_value=io.BytesIO(b"bad!")):
                    with self.assertRaisesRegex(ValueError, "integrity"):
                        audio.download_models()
            self.assertEqual(existing.read_bytes(), b"old!")
            self.assertFalse(list(Path(root).glob("*.part")))

    def test_music_ducks_under_speech(self):
        n = 4 * audio.RATE
        full = audio.music_bed(n, [])
        ducked = audio.music_bed(n, [{"type": "speech", "start": 0, "end": 4}])
        np.testing.assert_allclose(ducked, full * 0.25)
        self.assertTrue(np.isfinite(full).all())

    def test_existing_sidecar_blocks_output_before_model_load(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "lesson.wav"
            output.with_suffix(".srt").write_text("keep me")
            with patch.object(audio, "load_engine") as load:
                with self.assertRaisesRegex(ValueError, "already exists"):
                    audio.generate(audio.parse_script("Hello."), output)
                load.assert_not_called()
            self.assertEqual(output.with_suffix(".srt").read_text(), "keep me")


if __name__ == "__main__":
    unittest.main()
