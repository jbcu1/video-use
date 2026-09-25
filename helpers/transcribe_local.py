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
import bisect
import sys
import wave
from pathlib import Path

from transcribe import read_env_value


DEFAULT_WHISPER_MODEL = "large-v3-turbo"
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

# Silero settings for trimming word edges (not for choosing what Whisper hears):
# no padding, so the regions hug the speech, and split only on pauses of 200 ms
# or more, so a stop consonant's closure does not split a word.
SPEECH_VAD = {"min_silence_duration_ms": 200, "speech_pad_ms": 0}
# Silero opens a region 50-120 ms into a soft onset ("s", "sh", "f"), so a word
# may start this far ahead of its region before it counts as drifted.
SPEECH_MARGIN = 0.12
# Whisper also ends the last word before a pause early — "afternoon." by 0.4 s,
# which an 80 ms cut pad then clips. Such a word's end may grow this much toward
# the end of its speech region; more would swallow speech Whisper did not transcribe.
SPEECH_MAX_EXTEND = 0.5


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


def clamp_to_speech(
    words: list[dict],
    regions: list[tuple[float, float]],
    margin: float = SPEECH_MARGIN,
    max_extend: float = SPEECH_MAX_EXTEND,
) -> list[dict]:
    """Pull word edges out of the silence around them.

    Whisper's alignment starts the first word after a pause inside the pause —
    by 0.4 s with faster-whisper's VAD padding, by the whole pause with `small`
    — so the silence vanishes from `spacing`, pack_transcripts stops breaking
    phrases there and a cut at that word keeps dead air. Word ends are the
    more reliable edge, so each word is anchored to the speech region its end
    falls in: its end comes back to `margin` after that region, and its start
    moves up to `margin` before it when what the word covers ahead of the
    region is mostly silence. The last word of a region ends where the region
    does (by up to `max_extend`), since Whisper cuts it short. A word that
    overlaps no region (quiet speech the VAD missed) is left alone.

    `regions` are sorted, non-overlapping (start, end) speech spans from a VAD.
    """
    if not regions:
        return words
    starts = [r[0] for r in regions]
    anchored: list[tuple[int, float]] = []  # (word index, end of its region)
    for k, w in enumerate(words):
        ws, we = w["start"], w["end"]
        i = bisect.bisect_left(starts, we) - 1  # last region starting before the word ends
        if i < 0 or regions[i][1] <= ws:
            continue
        rs, re_ = regions[i]
        w["end"] = round(min(we, re_ + margin), 3)
        ahead = rs - ws
        if ahead > margin:
            speech, j = 0.0, i - 1
            while j >= 0 and regions[j][1] > ws:
                speech += regions[j][1] - max(regions[j][0], ws)
                j -= 1
            if speech < ahead / 2:
                w["start"] = round(rs - margin, 3)
        anchored.append((k, re_))
    # After the starts have moved: a word is its region's last when the next one starts past it.
    for k, re_ in anchored:
        w = words[k]
        nxt = words[k + 1]["start"] if k + 1 < len(words) else float("inf")
        if nxt >= re_ and w["end"] < re_:
            w["end"] = round(min(re_, w["end"] + max_extend), 3)
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

    def speech_regions(self, audio_path: Path) -> list[tuple[float, float]]:
        """(start, end) seconds of speech per Silero VAD, which faster-whisper ships."""
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        samples, sr = load_wav(audio_path)
        spans = get_speech_timestamps(samples, VadOptions(**SPEECH_VAD), sampling_rate=sr)
        return [(s["start"] / sr, s["end"] / sr) for s in spans]

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
        if words:
            clamp_to_speech(words, self.speech_regions(audio_path))

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
