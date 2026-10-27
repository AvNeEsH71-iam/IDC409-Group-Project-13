"""
Data preparation: build the audio subset, the noisy versions and the manifest.

Planned contents:
- download_librispeech_subset(): stream LibriSpeech dev-clean and keep a small, short-clip subset as 16 kHz wav
- add_noise(): add white noise at a chosen SNR (numpy)
- create_noisy_versions(): write noisy copies of every clean clip
- register_own_recordings(): optional - add your own (e.g. Indian-accented) recordings
- main(): save data/manifest.csv, fill the SQLite database, copy a few sample clips
Run with: python -m src.data_prep
"""
import argparse
import io
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from . import config, database


def _rel(path: Path) -> str:
    return Path(path).resolve().relative_to(config.ROOT).as_posix()


def download_librispeech_subset(n_clips=config.NUM_CLIPS,
                                max_per_speaker=config.MAX_PER_SPEAKER,
                                max_duration=config.MAX_DURATION_S) -> pd.DataFrame:
    """Stream the dev-clean archive and stop as soon as enough short clips are found."""
    config.ensure_dirs()
    transcripts, pending, per_speaker, rows = {}, {}, {}, []

    def resolve():
        for utt in list(pending):
            if utt in transcripts and len(rows) < n_clips:
                audio, sr = pending.pop(utt)
                path = config.AUDIO_CLEAN / f"{utt}.wav"
                sf.write(path, audio, sr, subtype="PCM_16")
                rows.append({
                    "clip_id": utt, "source_clip_id": utt, "filename": path.name,
                    "filepath": _rel(path), "condition": "clean", "snr_db": np.nan,
                    "duration_s": round(len(audio) / sr, 3), "sample_rate": sr,
                    "true_text": transcripts[utt].lower(),
                })

    print(f"Streaming {config.LIBRISPEECH_URL} (stops early once {n_clips} clips are found) ...")
    with urllib.request.urlopen(config.LIBRISPEECH_URL) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if name.endswith(".trans.txt"):
                    text = tar.extractfile(member).read().decode("utf-8")
                    for line in text.strip().splitlines():
                        utt, _, sentence = line.partition(" ")
                        transcripts[utt] = sentence
                    resolve()
                elif name.endswith(".flac"):
                    utt, speaker = name[:-5], name.split("-")[0]
                    if per_speaker.get(speaker, 0) >= max_per_speaker:
                        continue
                    audio, sr = sf.read(io.BytesIO(tar.extractfile(member).read()), dtype="float32")
                    if len(audio) / sr > max_duration:
                        continue
                    per_speaker[speaker] = per_speaker.get(speaker, 0) + 1
                    pending[utt] = (audio, sr)
                    resolve()
                if len(rows) >= n_clips:
                    break
    print(f"Saved {len(rows)} clean clips from {len(per_speaker)} speakers.")
    return pd.DataFrame(rows)


def add_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise so that signal-to-noise ratio equals snr_db."""
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape)
    noisy = signal + noise
    peak = np.max(np.abs(noisy))
    return (noisy / peak if peak > 1.0 else noisy).astype("float32")


def create_noisy_versions(clean_df: pd.DataFrame, snr_levels=config.SNR_LEVELS_DB) -> pd.DataFrame:
    rng = np.random.default_rng(config.RANDOM_SEED)
    rows = []
    for _, r in clean_df.iterrows():
        audio, sr = sf.read(config.ROOT / r["filepath"], dtype="float32")
        for snr in snr_levels:
            folder = config.AUDIO_NOISY / f"snr_{snr}dB"
            folder.mkdir(parents=True, exist_ok=True)
            cid = f"{r['clip_id']}_snr{snr}"
            path = folder / f"{cid}.wav"
            sf.write(path, add_noise(audio, snr, rng), sr, subtype="PCM_16")
            rows.append({**r.to_dict(), "clip_id": cid, "filename": path.name,
                         "filepath": _rel(path), "condition": f"noisy_{snr}dB", "snr_db": float(snr)})
    return pd.DataFrame(rows)


def register_own_recordings(csv_path, condition="accent_indian") -> pd.DataFrame:
    """CSV columns: filename, true_text. Audio files (wav/flac) sit next to the CSV."""
    csv_path = Path(csv_path)
    meta = pd.read_csv(csv_path)
    rows = []
    for _, r in meta.iterrows():
        path = csv_path.parent / r["filename"]
        info = sf.info(path)
        cid = f"own_{path.stem}"
        rows.append({"clip_id": cid, "source_clip_id": cid, "filename": path.name,
                     "filepath": _rel(path), "condition": condition, "snr_db": np.nan,
                     "duration_s": round(info.duration, 3), "sample_rate": info.samplerate,
                     "true_text": str(r["true_text"]).lower()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=config.NUM_CLIPS)
    ap.add_argument("--skip-download", action="store_true",
                    help="reuse the clean clips already listed in data/manifest.csv")
    ap.add_argument("--own-csv", help="optional CSV (filename,true_text) of your own recordings")
    args = ap.parse_args()

    config.ensure_dirs()
    if args.skip_download:
        clean = pd.read_csv(config.MANIFEST_CSV).query("condition == 'clean'")
    else:
        clean = download_librispeech_subset(args.n_clips)
    noisy = create_noisy_versions(clean)
    parts = [clean, noisy]
    if args.own_csv:
        parts.append(register_own_recordings(args.own_csv))
    manifest = pd.concat(parts, ignore_index=True)

    manifest.to_csv(config.MANIFEST_CSV, index=False)
    database.insert_clips(manifest)
    for _, r in clean.head(config.NUM_SAMPLE_CLIPS).iterrows():
        shutil.copy(config.ROOT / r["filepath"], config.SAMPLE_DIR / r["filename"])
    print(manifest.groupby("condition").size().rename("clips"))
    print(f"Manifest -> {config.MANIFEST_CSV}\nDatabase -> {config.DB_PATH}")


if __name__ == "__main__":
    main()
