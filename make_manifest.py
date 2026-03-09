import json
import argparse
import hashlib
import os
import subprocess

import numpy as np
import pandas as pd

from noise import add_noise_to_file


def get_md5(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_phonemes(text: str, lang: str) -> str:
    res = subprocess.run(
        ["espeak-ng", "-v", lang, "-q", "--ipa", text], capture_output=True, text=True
    )
    return res.stdout.strip()


def process_utterance(
    row: dict,
    lang: str,
    wav_dir: str,
    snr_db: float | None,
    noisy_wav_dir: str | None,
    seed: int | None,
) -> dict:
    src_wav = os.path.join(wav_dir, row["audio_file"])

    if snr_db is None:  # clean case
        out_wav = src_wav
    else:
        os.makedirs(noisy_wav_dir, exist_ok=True)
        noisy_filename = f"snr{snr_db}_{row['audio_file']}"
        out_wav = os.path.join(noisy_wav_dir, noisy_filename)

        add_noise_to_file(src_wav, out_wav, snr_db, seed)

    return {
        "utt_id": row["audio_id"],
        "lang": lang,
        "wav_path": out_wav,
        "ref_text": row["transcription"],
        "ref_phon": get_phonemes(row["transcription"], lang),
        "audio_md5": get_md5(out_wav),
        "sr": 16_000,
        "duration_s": row["duration_ms"] / 1_000,
        "snr_db": snr_db,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--wav-dir", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--snr-db", type=float, default=None)
    parser.add_argument("--noisy-wav-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.snr_db is not None and args.noisy_wav_dir is None:
        parser.error("--noisy-wav-dir is required when --snr-db is set")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    md = pd.read_csv(args.metadata, sep="\t")

    tmp_path = args.out + ".tmp"
    try:
        with open(tmp_path, "w", encoding="urf-8") as f:
            for _, row in md.iterrows():
                entry = process_utterance(
                    row=row.to_dict(),
                    lang=args.lang,
                    wav_dir=args.wav_dir,
                    snr_db=args.snr_db,
                    noisy_wav_dir=args.noisy_wav_dir,
                    seed=args.seed,
                )
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        os.replace(tmp_path, args.out)
        print(f"Manifest written to {args.out}")

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e
