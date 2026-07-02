import os
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

TARGET_SR = 16000

classifier: EncoderClassifier | None = None


def load_model():
    global classifier
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
    )


def load_audio(path: str) -> torch.Tensor:
    signal, sr = torchaudio.load(path, backend="soundfile")
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        signal = torchaudio.transforms.Resample(sr, TARGET_SR)(signal)
    return signal


def get_embedding(path: str) -> torch.Tensor:
    signal = load_audio(path)
    with torch.no_grad():
        embedding = classifier.encode_batch(signal)
    return embedding.squeeze()


def build_reference_embedding(audio_paths: list[str]) -> torch.Tensor:
    embeddings = [get_embedding(p) for p in audio_paths]
    if not embeddings:
        raise ValueError("Hech qanday audio fayl berilmadi!")
    return torch.stack(embeddings).mean(dim=0)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def verify_speaker(
    test_path: str,
    reference_embedding: torch.Tensor,
    threshold: float = 0.65,
) -> dict:
    test_emb = get_embedding(test_path)
    similarity = cosine_similarity(test_emb, reference_embedding)
    return {
        "similarity": round(similarity, 4),
        "threshold": threshold,
        "is_target": similarity >= threshold,
        "javob": "HA - bu siz suragan inson" if similarity >= threshold else "YO'Q - bu siz suragan odam emas",
    }