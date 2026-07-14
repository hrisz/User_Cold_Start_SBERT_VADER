import os
import shutil

os.environ["HF_HUB_DISABLE_XET"] = "1"

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"
LOCAL_MODEL_DIR = "model/sbert_model"

if os.path.exists(LOCAL_MODEL_DIR):
    shutil.rmtree(LOCAL_MODEL_DIR)
    print("Folder model lama dihapus:", LOCAL_MODEL_DIR)

print("Download model SBERT...")
model = SentenceTransformer(MODEL_NAME)

print("Simpan model ke:", LOCAL_MODEL_DIR)
model.save(LOCAL_MODEL_DIR)

print("Selesai. Model SBERT berhasil disimpan lokal.")