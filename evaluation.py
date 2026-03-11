import torch
import torchaudio
import os
import json
import argparse
import Levenshtein

import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


def load_audio(wav_path: str, target_sr: int = 16000) -> np.ndarray:
    waveform, sr = torchaudio.load(wav_path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze().numpy()


def inference_batch(
    rows: list[dict],
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
) -> list[dict]:
    waveforms = [load_audio(row["wav_path"]) for row in rows]

    inputs = processor(
        waveforms,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcriptions = processor.batch_decode(predicted_ids)

    return [
        {**row, "hyp_phon": hyp}
        for row, hyp in zip(rows, transcriptions)
    ]


def compute_per(rows: list[dict]) -> float:
    """Corpus-level PER: sum of edit distances / sum of ref lengths."""
    total_dist = sum(
        Levenshtein.distance(
            row["ref_phon"].split(),
            row["hyp_phon"].split(),
        )
        for row in rows
    )
    total_ref = sum(len(row["ref_phon"].split()) for row in rows)
    return total_dist / total_ref if total_ref > 0 else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",    required=True)
    parser.add_argument("--out-pred",    required=True)
    parser.add_argument("--out-metrics", required=True)
    parser.add_argument("--batch-size",  type=int, default=16)
    args = parser.parse_args()

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    model     = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    model.eval()

    with open(args.manifest, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    predicted_rows = []
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i : i + args.batch_size]
        predicted_rows.extend(inference_batch(batch, processor, model))
        print(f"Processed {min(i + args.batch_size, len(rows))}/{len(rows)}")

    os.makedirs(os.path.dirname(args.out_pred), exist_ok=True)
    tmp = args.out_pred + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for row in predicted_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, args.out_pred)

    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise e


    per = compute_per(predicted_rows)
    snr = predicted_rows[0].get("snr_db")
    lang = predicted_rows[0].get("lang")

    metrics = {"lang": lang, "snr_db": snr, "per": per}
    print(f"PER ({lang}, SNR={snr}): {per:.4f}")

    os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
    with open(args.out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
