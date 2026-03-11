import argparse
import subprocess
import sys
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",       required=True)
    parser.add_argument("--out-dir",        required=True)
    parser.add_argument("--noisy-wav-dir",  required=True)
    args = parser.parse_args()

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    for snr in params["snr_levels"]:
        subprocess.run([
            sys.executable, "make_manifest.py",
            "--manifest",      args.manifest,
            "--out",           f"{args.out_dir}/snr{snr}.jsonl",
            "--snr-db",        str(snr),
            "--noisy-wav-dir", f"{args.noisy_wav_dir}/snr{snr}/",
            "--seed",          str(params["seed"]),
        ], check=True)