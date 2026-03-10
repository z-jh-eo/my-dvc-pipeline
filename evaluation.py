import torch
import torchaudio
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")


def inference(file_paths, processor, model):
    audio_arrays = []
    for path in file_paths:
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        audio_arrays.append(waveform.squeeze().numpy())

    inputs = processor(
        audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)

