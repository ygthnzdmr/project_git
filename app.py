from flask import Flask, render_template, request, redirect, url_for, send_file
from pathlib import Path
import torch, threading, random, time, os
from fastai.vision.all import *

# =========================
# CONFIG
# =========================
BASE = Path("DataSets")
TRAIN_DIR = BASE / "Chess"
TEST_DIR  = BASE / "Test"

MODEL_PATH = TRAIN_DIR / "chess_piece_class.pkl"


UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.70
last_accuracy = None
model_ready = False
# =========================
# DEVICE
# =========================
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# =========================
# MODEL LOAD / CREATE
# =========================
def load_or_create_model():
    global model_ready
    if MODEL_PATH.exists():
        print("✅ Kayıtlı model yüklendi")
        learn = load_learner(MODEL_PATH)
        model_ready = True
    else:
        print("⚠️ Model yok, yeni model oluşturuluyor")

        dls = ImageDataLoaders.from_folder(
            TRAIN_DIR,
            valid_pct=0.2,
            seed=42,
            item_tfms=Resize(224),
            bs=16
        )

        learn = cnn_learner(
            dls,
            resnet34,
            metrics=accuracy
        )
        model_ready = False

    learn.dls.device = device
    learn.model.to(device)
    return learn

learn = load_or_create_model()

# =========================
# FLASK
# =========================
app = Flask(__name__)

# -------------------------
# TRAIN
# -------------------------
@app.route("/train", methods=["POST"])
def train():
    global learn, last_accuracy, model_ready
    epochs = int(request.form.get("epochs", 5))

    def bg_train():
        global learn, last_accuracy, model_ready
        print("🚀 Eğitim başladı")

        learn.fit(epochs)
        preds, targs = learn.get_preds()
        last_accuracy = (preds.argmax(dim=1) == targs).float().mean().item()


        # klasörü garanti altına al
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        learn.path = TRAIN_DIR
        # ⚠️ SADECE DOSYA ADI
        learn.export(MODEL_PATH.name)
        model_ready = True

        print(f"✅ Eğitim bitti & model kaydedildi → {MODEL_PATH}")

    threading.Thread(target=bg_train, daemon=True).start()
    return "", 204

# -------------------------
# TEST IMAGE SERVE
# -------------------------

@app.route("/predict", methods=["POST"])
def predict_api():
    # Model eğitilmediyse net hata
    if not training_state["trained"]:
        return jsonify({"error": "Model henüz eğitilmedi"}), 400

    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify({"error": "image field boş"}), 400

    # Unique isim (aynı isim çakışmasın)
    ext = Path(file.filename).suffix or ".jpg"
    fname = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / fname
    file.save(save_path)

    # Tahmin
    img = PILImage.create(save_path)
    pred, idx, probs = learn.predict(img)
    conf = float(probs[idx].item())          # 0..1

    label = str(pred) if conf >= CONF_THRESHOLD else "Emin değilim"
    img_url = url_for("static", filename=f"uploads/{fname}", _external=True)

    # iOS yüzde basıyor diye yüzde gönderiyorum
    return jsonify({
        "label": label,
        "confidence": conf * 100.0,          # 0..100
        "additionalInfo": img_url
    })

@app.route("/test_image/<name>")
def test_image(name):
    return send_file(TEST_DIR / name)

# -------------------------
# INDEX
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    img_path = None
    result = None
    confidence = None

    graphs = {
        "learning": os.path.exists("static/learning_curve.png"),
        "confusion": os.path.exists("static/confusion_matrix.png"),
        "roc": os.path.exists("static/roc_curve.png")
    }

    # 🔒 MODEL YOKSA POST ENGELLE
    if request.method == "POST" and not model_ready:
        return render_template(
            "index.html",
            trained=True,
            img_path=None,
            result=None,
            confidence=None,
            accuracy=last_accuracy,
            graphs=graphs,
            model_ready=False
        )

    if request.method == "POST":

        if "random_test" in request.form:
            img_name = random.choice(os.listdir(TEST_DIR))
            img = PILImage.create(TEST_DIR / img_name)

            pred, idx, probs = learn.predict(img)
            confidence = probs[idx].item()

            result = str(pred) if confidence >= CONF_THRESHOLD else "Emin değilim"
            img_path = url_for("test_image", name=img_name)

        elif "image" in request.files:
            file = request.files["image"]
            if file and file.filename:
                save_path = UPLOAD_DIR / file.filename
                file.save(save_path)

                img = PILImage.create(save_path)
                pred, idx, probs = learn.predict(img)

                confidence = probs[idx].item()
                result = str(pred) if confidence >= CONF_THRESHOLD else "Emin değilim"
                img_path = url_for("static", filename=f"uploads/{file.filename}")

    return render_template(
        "index.html",
        trained=True,
        img_path=img_path,
        result=result,
        confidence=confidence,
        accuracy=last_accuracy,
        graphs=graphs,
        model_ready=model_ready
    )



# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
