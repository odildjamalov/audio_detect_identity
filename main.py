import os
import tempfile
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile

import speaker
from config.settings import settings

embeddings_cache: dict[str, torch.Tensor] = {}


def embedding_path(name: str) -> str:
    return os.path.join(settings.reference_embeddings_dir, f"{name}.pt")


def load_all_embeddings():
    d = settings.reference_embeddings_dir
    if not os.path.exists(d):
        return
    for fname in os.listdir(d):
        if fname.endswith(".pt"):
            name = fname[:-3]
            embeddings_cache[name] = torch.load(
                os.path.join(d, fname), weights_only=True
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    speaker.load_model()
    load_all_embeddings()
    yield


app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)


@app.get("/status")
def status():
    return {
        "model_loaded": speaker.classifier is not None,
        "enrolled_speakers": list(embeddings_cache.keys()),
    }


@app.post("/enroll")
async def enroll(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
):

    saved_paths = []
    try:
        for upload in files:
            suffix = os.path.splitext(upload.filename)[-1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await upload.read())
                saved_paths.append(tmp.name)

        embedding = speaker.build_reference_embedding(saved_paths)
    finally:
        for p in saved_paths:
            os.unlink(p)

    os.makedirs(settings.reference_embeddings_dir, exist_ok=True)
    torch.save(embedding, embedding_path(name))
    embeddings_cache[name] = embedding

    return {
        "message": f"'{name}' uchun embedding yaratildi va saqlandi",
        "files_used": len(saved_paths),
        "saved_to": embedding_path(name),
    }


@app.post("/verify")
async def verify(
    name: str = Form(...),
    file: UploadFile = File(...),
    threshold: float = Query(default=settings.threshold, ge=0.0, le=1.0),
):
    """
    Yuklangan audio berilgan speaker (name) ga tegishli yoki yo'qligini tekshiradi.
    """
    if name not in embeddings_cache:
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' nomli speaker topilmadi. Avval /enroll qiling.",
        )

    suffix = os.path.splitext(file.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = speaker.verify_speaker(tmp_path, embeddings_cache[name], threshold)
    finally:
        os.unlink(tmp_path)

    return {"speaker": name, **result}


@app.delete("/speakers/{name}")
def delete_speaker(name: str):
    if name not in embeddings_cache:
        raise HTTPException(status_code=404, detail=f"'{name}' topilmadi.")

    os.remove(embedding_path(name))
    del embeddings_cache[name]

    return {"message": f"'{name}' o'chirildi."}
