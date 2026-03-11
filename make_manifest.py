import json
import argparse
import hashlib
import os
import subprocess

from noise import add_noise_to_file


def get_md5(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def process_utterance(
        json_row: dict,
        snr_db: float,
        noisy_wav_dir: str,
        seed: int
) -> dict:
    
    os.makedirs(noisy_wav_dir, exist_ok=True)
    noisy_filename = f"snr{snr_db}_{os.path.basename(json_row['wav_path'])}"
    out_wav = os.path.join(noisy_wav_dir, noisy_filename)

    add_noise_to_file(json_row["wav_path"], out_wav, snr_db, seed)

    return {
        **json_row,
        "wav_path":  out_wav,
        "audio_md5": get_md5(out_wav),
        "snr_db":    snr_db,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",      required=True)
    parser.add_argument("--out",           required=True)
    parser.add_argument("--snr-db",        type=float, default=None)
    parser.add_argument("--noisy-wav-dir", default=None)
    parser.add_argument("--seed",          type=int, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp_path = args.out + ".tmp"

    try:
        with open(args.manifest, "r", encoding="utf-8") as f:
            json_rows = [json.loads(line) for line in f]
        noisy_rows = [
            process_utterance(row, args.snr_db, args.noisy_wav_dir, args.seed)
            for row in json_rows
        ]

        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in noisy_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        
        os.replace(tmp_path, args.out)
        print(f"Manifest written to {args.out}")



    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e
