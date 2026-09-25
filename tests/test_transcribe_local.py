import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


HELPERS = Path(__file__).parents[1] / "helpers"
sys.path.insert(0, str(HELPERS))  # transcribe_local imports its sibling `transcribe`

import pack_transcripts  # noqa: E402
import transcribe  # noqa: E402
import transcribe_local  # noqa: E402


def word(text, start, end, **extra):
    return {"text": text, "start": start, "end": end, "type": "word", **extra}


class AssignSpeakersTests(unittest.TestCase):
    def test_picks_turn_with_most_overlap_and_renames_in_order_of_appearance(self):
        words = [word("hi", 0.0, 0.4), word("there", 0.9, 1.3), word("yo", 2.0, 2.2)]
        turns = [(1.0, 3.0, "SPEAKER_07"), (0.0, 1.0, "SPEAKER_03")]
        transcribe_local.assign_speakers(words, turns)
        # "there" overlaps SPEAKER_03 by 0.1s and SPEAKER_07 by 0.3s.
        self.assertEqual([w["speaker_id"] for w in words],
                         ["speaker_0", "speaker_1", "speaker_1"])

    def test_word_outside_every_turn_takes_the_nearest(self):
        words = [word("late", 5.0, 5.2), word("early", 0.1, 0.2)]
        turns = [(1.0, 2.0, "A"), (4.0, 4.5, "B")]
        transcribe_local.assign_speakers(words, turns)
        self.assertEqual([w["speaker_id"] for w in words], ["speaker_1", "speaker_0"])

    def test_no_turns_leaves_words_unlabelled(self):
        words = [word("hi", 0.0, 0.4)]
        transcribe_local.assign_speakers(words, [])
        self.assertNotIn("speaker_id", words[0])


class ScribeShapeTests(unittest.TestCase):
    def test_spacing_between_every_pair_carries_the_gap_and_speaker(self):
        words = [word("a", 0.0, 0.5, speaker_id="speaker_0"),
                 word("b", 1.2, 1.4, speaker_id="speaker_1")]
        out = transcribe_local.with_spacing(words)
        self.assertEqual([w["type"] for w in out], ["word", "spacing", "word"])
        self.assertEqual((out[1]["start"], out[1]["end"]), (0.5, 1.2))
        self.assertEqual(out[1]["speaker_id"], "speaker_0")

    def test_overlapping_words_give_zero_length_spacing(self):
        out = transcribe_local.with_spacing([word("a", 0.0, 0.6), word("b", 0.5, 0.9)])
        self.assertEqual((out[1]["start"], out[1]["end"]), (0.6, 0.6))
        self.assertNotIn("speaker_id", out[1])

    def test_whisper_words_strips_skips_blanks_and_clamps_end(self):
        segments = [SimpleNamespace(words=[
            SimpleNamespace(word=" Hello,", start=0.1234, end=0.5, probability=0.91),
            SimpleNamespace(word=" ", start=0.5, end=0.6, probability=0.1),
            SimpleNamespace(word=" world", start=0.7, end=0.69, probability=0.8),
        ]), SimpleNamespace(words=None)]
        words = transcribe_local.whisper_words(segments)
        self.assertEqual([w["text"] for w in words], ["Hello,", "world"])
        self.assertEqual(words[0]["start"], 0.123)
        self.assertEqual(words[1]["end"], words[1]["start"])


class BackendTests(unittest.TestCase):
    def test_auto_prefers_scribe_only_when_a_key_resolves(self):
        with patch.object(transcribe, "read_env_value", return_value="sk-test"):
            self.assertEqual(transcribe.resolve_backend("auto"), "elevenlabs")
        with patch.object(transcribe, "read_env_value", return_value=""):
            self.assertEqual(transcribe.resolve_backend("auto"), "local")
        self.assertEqual(transcribe.resolve_backend("local"), "local")

    def test_empty_value_in_dotenv_falls_through_to_environment(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, ".env").write_text("VIDEO_USE_TEST_TOKEN=\n")
            os.chdir(tmp)
            try:
                with patch.dict(os.environ, {"VIDEO_USE_TEST_TOKEN": "from-env"}):
                    self.assertEqual(transcribe.read_env_value("VIDEO_USE_TEST_TOKEN"), "from-env")
                Path(tmp, ".env").write_text("VIDEO_USE_TEST_TOKEN='from-file'\n")
                self.assertEqual(transcribe.read_env_value("VIDEO_USE_TEST_TOKEN"), "from-file")
            finally:
                os.chdir(old_cwd)


class FakeWhisper:
    """Stands in for faster_whisper.WhisperModel: two words, then two more."""

    def __init__(self):
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        seg = lambda *ws: SimpleNamespace(words=[  # noqa: E731
            SimpleNamespace(word=f" {t}", start=s, end=e, probability=0.9) for t, s, e in ws])
        segments = iter([seg(("Hello", 0.2, 0.5), ("there.", 0.55, 0.9)),
                         seg(("Hi", 1.8, 2.0), ("back.", 2.05, 2.4))])
        return segments, SimpleNamespace(language="en", language_probability=0.99)


class FakeTranscriber(transcribe_local.LocalTranscriber):
    """LocalTranscriber with the models swapped out; diarize() returns fixed turns."""

    def __init__(self, turns=None):
        self.model = FakeWhisper()
        self.model_name = "fake"
        self.prompt = "Umm, uh."
        self.verbose = False
        self.diarizer = object() if turns is not None else None
        self.turns = turns

    def diarize(self, audio_path, num_speakers):
        return self.turns


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
class TranscribeOneLocalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.video = self.tmp / "take1.mp4"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "color=c=black:s=160x90:d=3",
             "-f", "lavfi", "-i", "sine=frequency=300:duration=3",
             "-c:v", "libx264", "-c:a", "aac", "-shortest", str(self.video)],
            check=True,
        )
        self.edit = self.tmp / "edit"

    def run_engine(self, engine, **kwargs):
        out = transcribe.transcribe_one(self.video, self.edit, None, verbose=False,
                                        engine=engine, **kwargs)
        return out, json.loads(out.read_text())

    def test_writes_scribe_shaped_transcript_that_packs_by_speaker(self):
        engine = FakeTranscriber(turns=[(0.0, 1.0, "SPEAKER_01"), (1.5, 2.6, "SPEAKER_00")])
        out, data = self.run_engine(engine, language="en")

        self.assertEqual(out, self.edit / "transcripts" / "take1.json")
        path, kwargs = engine.model.calls[0]
        self.assertTrue(path.endswith(".wav"))
        self.assertEqual(kwargs["language"], "en")
        self.assertTrue(kwargs["word_timestamps"])
        self.assertEqual(kwargs["initial_prompt"], "Umm, uh.")
        self.assertEqual(data["transcriber"]["diarization"], transcribe_local.DIARIZATION_MODEL)

        phrases = pack_transcripts.group_into_phrases(data["words"], silence_threshold=5.0)
        self.assertEqual([(p["speaker_id"], p["text"]) for p in phrases],
                         [("speaker_0", "Hello there."), ("speaker_1", "Hi back.")])

    def test_without_diarizer_words_have_no_speaker_and_split_on_silence(self):
        _, data = self.run_engine(FakeTranscriber(turns=None))
        self.assertIsNone(data["transcriber"]["diarization"])
        self.assertFalse(any("speaker_id" in w for w in data["words"]))
        phrases = pack_transcripts.group_into_phrases(data["words"], silence_threshold=0.5)
        self.assertEqual([p["text"] for p in phrases], ["Hello there.", "Hi back."])

    def test_one_known_speaker_skips_diarization(self):
        engine = FakeTranscriber(turns=[(0.0, 1.0, "A"), (1.5, 2.6, "B")])
        _, data = self.run_engine(engine, num_speakers=1)
        self.assertEqual({w.get("speaker_id") for w in data["words"]}, {"speaker_0"})
        self.assertIsNone(data["transcriber"]["diarization"])

    def test_cached_transcript_is_not_retranscribed(self):
        engine = FakeTranscriber()
        self.run_engine(engine)
        self.run_engine(engine)
        self.assertEqual(len(engine.model.calls), 1)


@unittest.skipUnless(importlib.util.find_spec("pyannote.core"), "pyannote.audio not installed")
class PyannoteOutputTests(unittest.TestCase):
    def test_reads_the_exclusive_view_of_pyannote_4_output(self):
        from pyannote.core import Annotation, Segment

        def annotation(*turns):
            a = Annotation()
            for s, e, label in turns:
                a[Segment(s, e)] = label
            return a

        output = SimpleNamespace(
            speaker_diarization=annotation((0.0, 2.0, "A"), (1.0, 3.0, "B")),
            exclusive_speaker_diarization=annotation((0.0, 1.0, "A"), (1.0, 3.0, "B")),
        )
        engine = FakeTranscriber()
        engine.diarizer = lambda audio, **kw: output
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "a.wav"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi",
                            "-i", "sine=duration=1", "-ac", "1", "-ar", "16000",
                            "-c:a", "pcm_s16le", str(wav)], check=True)
            turns = transcribe_local.LocalTranscriber.diarize(engine, wav, None)
        self.assertEqual(turns, [(0.0, 1.0, "A"), (1.0, 3.0, "B")])


if __name__ == "__main__":
    unittest.main()
