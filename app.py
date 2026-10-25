import os
import uuid
import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, flash
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.keras")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "ganti-kunci-rahasia-untuk-deployment")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

# Penting: urutan ini mengikuti class_indices dari flow_from_directory
# yang mengurutkan nama folder secara alfabetis.
CLASS_NAMES = [
    "BELIMBING BUSUK",
    "BELIMBING SEGAR",
    "BELIMBING TIDAK SEGAR",
]
CLASS_META = {
    "BELIMBING SEGAR": {
        "title": "Segar", "tone": "fresh", "icon": "✦",
        "description": "Citra paling sesuai dengan kelas belimbing segar.",
        "advice": "Periksa juga aroma, tekstur, dan kondisi buah secara langsung sebelum dikonsumsi."
    },
    "BELIMBING TIDAK SEGAR": {
        "title": "Tidak Segar", "tone": "warning", "icon": "◷",
        "description": "Citra paling sesuai dengan kelas belimbing tidak segar.",
        "advice": "Periksa perubahan tekstur, aroma, dan bagian buah yang mulai layu atau berubah warna."
    },
    "BELIMBING BUSUK": {
        "title": "Busuk", "tone": "rotten", "icon": "!",
        "description": "Citra paling sesuai dengan kelas belimbing busuk.",
        "advice": "Jangan hanya mengandalkan hasil citra; periksa kondisi buah dan hindari konsumsi bila menunjukkan tanda pembusukan."
    },
}

model = None
model_error = None
try:
    if os.path.isfile(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    else:
        model_error = "File best_model.keras tidak ditemukan."
except Exception as exc:
    model_error = f"Model gagal dimuat: {exc}"

def predict_image(image_path):
    if model is None:
        raise RuntimeError(model_error or "Model belum siap.")
    with Image.open(image_path) as im:
        im = im.convert("RGB").resize((224, 224))
        arr = np.asarray(im, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    scores = np.asarray(model.predict(arr, verbose=0))[0].astype(float).ravel()
    if len(scores) != len(CLASS_NAMES):
        raise RuntimeError(
            f"Model menghasilkan {len(scores)} keluaran, sedangkan konfigurasi kelas berisi {len(CLASS_NAMES)} kelas. "
            "Periksa kembali model dan urutan label."
        )
    idx = int(np.argmax(scores))
    predicted = CLASS_NAMES[idx]
    probs = [
        {"name": name, "title": CLASS_META[name]["title"], "value": float(scores[i]) * 100}
        for i, name in enumerate(CLASS_NAMES)
    ]
    probs.sort(key=lambda x: x["value"], reverse=True)
    return predicted, float(scores[idx]) * 100, probs

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_url = None
    if request.method == "POST":
        if model is None:
            flash("Model belum siap. " + (model_error or "Periksa file model dan instalasi TensorFlow."), "error")
            return render_template("index.html", result=None, image_url=None, model_ready=False)
        uploaded = request.files.get("fruit_image")
        if not uploaded or not uploaded.filename:
            flash("Pilih foto belimbing terlebih dahulu.", "error")
            return render_template("index.html", result=None, image_url=None, model_ready=True)
        try:
            # Validasi format dengan Pillow, bukan hanya ekstensi nama file.
            img = Image.open(uploaded.stream)
            img.verify()
            uploaded.stream.seek(0)
            img = Image.open(uploaded.stream).convert("RGB")
            filename = f"{uuid.uuid4().hex}.jpg"
            saved_path = os.path.join(UPLOAD_DIR, filename)
            img.save(saved_path, format="JPEG", quality=92, optimize=True)
            predicted, confidence, probabilities = predict_image(saved_path)
            meta = CLASS_META[predicted]
            result = {
                "label": predicted,
                "title": meta["title"],
                "tone": meta["tone"],
                "icon": meta["icon"],
                "description": meta["description"],
                "advice": meta["advice"],
                "confidence": confidence,
                "probabilities": probabilities,
            }
            image_url = "/" + os.path.relpath(saved_path, BASE_DIR).replace(os.sep, "/")
        except (UnidentifiedImageError, OSError, ValueError):
            flash("File tidak dapat dibaca sebagai gambar. Gunakan JPG, JPEG, atau PNG.", "error")
        except Exception as exc:
            app.logger.exception("Prediksi gagal")
            flash(f"Prediksi gagal diproses. Periksa kompatibilitas model. Detail: {exc}", "error")
    return render_template("index.html", result=result, image_url=image_url, model_ready=(model is not None))

@app.errorhandler(413)
def too_large(_):
    flash("Ukuran foto terlalu besar. Maksimum 8 MB.", "error")
    return render_template("index.html", result=None, image_url=None, model_ready=(model is not None)), 413

if __name__ == "__main__":
    # Untuk pengujian lokal. Jangan gunakan debug=True pada server publik.
    app.run(host="127.0.0.1", port=5000, debug=False)
