"""Local transcription: faster-whisper for words, pyannote for speakers.

The free, on-device alternative to ElevenLabs Scribe, used by transcribe.py and
transcribe_batch.py with `--backend local`. It writes the shape Scribe returns
— a `words` list of `word` and `spacing` entries with start, end and
speaker_id — so pack_transcripts.py, timeline_view.py and render.py read it
unchanged.

What it does not match:
  - No audio events. Whisper does not tag `(laughter)` or `(applause)`.
  - Filler words. Whisper tends to drop "umm" / "uh" and smooth false starts.
    `--whisper-prompt` with a filler-laden sentence in the footage's language
    brings most of them back; otherwise look for them in the waveform.
  - Speed. On CPU expect roughly real time for large-v3-turbo; `small` is
    several times faster and less accurate.

Diarization runs when pyannote.audio is installed and HF_TOKEN resolves (env
or .env). The pyannote model is gated: accept its terms on huggingface.co once,
with the account that owns the token. Without it every word gets no
speaker_id, and speaker changes no longer split phrases.

Install: `uv sync --extra local` (Whisper only) or
`uv sync --extra local --extra diarize` (Whisper + pyannote).
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

from transcribe import read_env_value


DEFAULT_WHISPER_MODEL = "large-v3-turbo"
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


# -------- Scribe-shaped output ------------------------------------------------


def assign_speakers(
    words: list[dict],
    turns: list[tuple[float, float, str]],
) -> list[dict]:
    """Label each word with the speaker whose turn overlaps it most.

    `turns` are (start, end, label) from the diarizer. Labels are renamed to
    Scribe's `speaker_N`, numbered in order of first appearance. A word that
    overlaps no turn (Whisper and pyannote disagree about where speech is)
    takes the nearest one.
    """
    if not turns:
        return words
    turns = sorted(turns)
    names: dict[str, str] = {}
    for _, _, label in turns:
        names.setdefault(label, f"speaker_{len(names)}")

    for w in words:
        ws, we = w["start"], w["end"]
        best, best_overlap = None, 0.0
        for ts, te, label in turns:
            overlap = min(we, te) - max(ws, ts)
            if overlap > best_overlap:
                best, best_overlap = label, overlap
        if best is None:
            mid = (ws + we) / 2
            best = min(turns, key=lambda t: max(t[0] - mid, mid - t[1], 0.0))[2]
        w["speaker_id"] = names[best]
    return words


def with_spacing(words: list[dict]) -> list[dict]:
    """Interleave Scribe-style `spacing` entries between consecutive words.

    pack_transcripts.py breaks phrases on long spacing entries and
    timeline_view.py shades them as silence.
    """
    out: list[dict] = []
    for i, w in enumerate(words):
        if i:
            prev = words[i - 1]
            gap = {"text": " ", "start": prev["end"], "end": max(prev["end"], w["start"]),
                   "type": "spacing"}
            if "speaker_id" in prev:
                gap["speaker_id"] = prev["speaker_id"]
            out.append(gap)
        out.append(w)
    return out


def whisper_words(segments) -> list[dict]:
    """Flatten faster-whisper segments into Scribe `word` entries."""
    words: list[dict] = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if not text:
                continue
            start = round(float(w.start), 3)
            end = round(max(float(w.end), start), 3)
            words.append({"text": text, "start": start, "end": end, "type": "word",
                          "probability": round(float(w.probability), 4)})
    return words


# -------- Models ---------------------------------------------------------------


def _cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def load_wav(path: Path):
    """16-bit mono wav → (float32 numpy array in [-1, 1], sample_rate)."""
    import numpy as np
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return samples.astype(np.float32) / 32768.0, sr


class LocalTranscriber:
    """Loads the models once; call it per audio file like transcribe.Engine."""

    def __init__(
        self,
        model: str = DEFAULT_WHISPER_MODEL,
        diarize: bool | None = None,
        prompt: str | None = None,
        verbose: bool = True,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            sys.exit("faster-whisper is not installed. Run `uv sync --extra local` "
                     "(or `pip install -e '.[local]'`) in the video-use repo.")

        device, compute_type = ("cuda", "float16") if _cuda_available() else ("cpu", "int8")
        if verbose:
            print(f"  loading whisper {model} on {device} ({compute_type})", flush=True)
        try:
            self.model = WhisperModel(model, device=device, compute_type=compute_type)
        except Exception as e:
            # First use downloads the weights from huggingface.co.
            sys.exit(f"could not load whisper model {model!r}: {type(e).__name__}: {e}. "
                     "The first run downloads it from huggingface.co — check network access.")
        self.model_name = model
        self.prompt = prompt
        self.verbose = verbose
        self.diarizer = self._load_diarizer(diarize)

    def _load_diarizer(self, diarize: bool | None):
        """None = diarize if possible; True = must diarize; False = don't."""
        if diarize is False:
            return None

        def unavailable(reason: str):
            if diarize:
                sys.exit(f"diarization requested but {reason}")
            if self.verbose:
                print(f"  note: no speaker diarization — {reason}", flush=True)
            return None

        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError:
            return unavailable("pyannote.audio is not installed (`uv sync --extra diarize`)")
        token = read_env_value("HF_TOKEN")
        if not token:
            return unavailable("HF_TOKEN not found in .env or environment")

        if self.verbose:
            print(f"  loading {DIARIZATION_MODEL}", flush=True)
        try:
            pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
        except Exception as e:
            return unavailable(
                f"{DIARIZATION_MODEL} failed to load ({e}). Accept its terms at "
                f"https://huggingface.co/{DIARIZATION_MODEL} with the HF_TOKEN account.")
        if pipeline is None:
            return unavailable(
                f"{DIARIZATION_MODEL} is gated — accept its terms at "
                f"https://huggingface.co/{DIARIZATION_MODEL} with the HF_TOKEN account.")
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        return pipeline

    def diarize(self, audio_path: Path, num_speakers: int | None) -> list[tuple[float, float, str]]:
        import torch
        samples, sr = load_wav(audio_path)
        # Hand pyannote the waveform itself, so it needs no audio decoder of its own.
        audio = {"waveform": torch.from_numpy(samples).unsqueeze(0), "sample_rate": sr}
        kwargs = {"num_speakers": num_speakers} if num_speakers else {}
        output = self.diarizer(audio, **kwargs)
        # pyannote 4 returns both views; the exclusive one (one speaker at a time)
        # is the one meant for aligning with a transcript.
        if hasattr(output, "exclusive_speaker_diarization"):
            annotation = output.exclusive_speaker_diarization
        else:
            annotation = getattr(output, "speaker_diarization", output)
        return [(float(seg.start), float(seg.end), str(label))
                for seg, _, label in annotation.itertracks(yield_label=True)]

    def __call__(self, audio_path: Path, language: str | None, num_speakers: int | None) -> dict:
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,
            initial_prompt=self.prompt,
        )
        words = whisper_words(segments)

        diarized = False
        if self.diarizer is not None and num_speakers != 1 and words:
            assign_speakers(words, self.diarize(audio_path, num_speakers))
            diarized = True
        elif num_speakers == 1:
            for w in words:
                w["speaker_id"] = "speaker_0"

        return {
            "language_code": info.language,
            "language_probability": round(float(info.language_probability), 4),
            "text": " ".join(w["text"] for w in words),
            "words": with_spacing(words),
            "transcriber": {
                "backend": "local",
                "whisper_model": self.model_name,
                "diarization": DIARIZATION_MODEL if diarized else None,
            },
        }


# -------- CLI glue -------------------------------------------------------------


def add_local_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("local backend (--backend local)")
    g.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        help=f"faster-whisper model name or path (default {DEFAULT_WHISPER_MODEL}; "
             "`small` is much faster on CPU, less accurate).",
    )
    g.add_argument(
        "--whisper-prompt",
        default=None,
        help="Initial prompt for Whisper. A sentence full of fillers in the footage's "
             "language (\"Umm, so, uh, like...\") keeps Whisper from dropping them.",
    )
    g.add_argument(
        "--diarize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Speaker diarization with pyannote. Default: on when pyannote.audio is "
             "installed and HF_TOKEN is set. --diarize makes it required.",
    )


def make_local_transcriber(args: argparse.Namespace, num_speakers: int | None,
                           verbose: bool = True) -> LocalTranscriber:
    # One known speaker needs no diarization; skip loading pyannote for it.
    diarize = False if num_speakers == 1 else args.diarize
    return LocalTranscriber(model=args.whisper_model, diarize=diarize,
                            prompt=args.whisper_prompt, verbose=verbose)
