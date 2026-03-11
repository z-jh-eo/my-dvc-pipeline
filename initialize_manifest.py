import json
import argparse
import hashlib
import os
import subprocess

import pandas as pd

def get_md5(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_phonemes(text: str, lang: str) -> str:
    if not text or (isinstance(text, float)):
        return ""
    res = subprocess.run(
        ["espeak-ng", "-v", lang, "-q", "--ipa", text],
        capture_output=True,
        text=True
    )
    return res.stdout.strip()


def ensure_wav(src_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(src_path))[0]
    wav_path = os.path.join(out_dir, stem + ".wav")
    if not os.path.exists(wav_path):
        subprocess.run(
            ["ffmpeg", "-i", src_path, "-ar", "16000", "-ac", "1", wav_path],
            check=True,
            capture_output=True,
        )
    return wav_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--wav-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    md = pd.read_csv(args.metadata, sep="\t")

    tmp_path = args.out + ".tmp"

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for _, row in md.iterrows():

                src_wav = ensure_wav(
                    os.path.join(args.wav_dir, row["audio_file"]),
                    out_dir=args.wav_dir.rstrip("/") + "_wav/"
                )

                src_stem = os.path.splitext(os.path.basename(src_wav))[0]

                transcription = row.get("transcription", "")
                transcription = "" if not isinstance(transcription, str) else transcription

                entry = {
                    "utt_id": args.lang + "_" + src_stem,
                    "lang": args.lang,
                    "wav_path": src_wav,
                    "ref_text": transcription,
                    "ref_phon": get_phonemes(transcription, args.lang),
                    "audio_md5": get_md5(src_wav),
                    "sr": 16_000,
                    "duration_s": row["duration_ms"] / 1_000,
                    "snr_db": None
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        os.replace(tmp_path, args.out)
        print(f"Manifest written to {args.out}")

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e
