from flask import Flask, render_template, request, send_file, redirect, url_for, jsonify
from pathlib import Path
import torch
import threading
import random
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")   # GUI olmayan backend

from fastai.vision.all import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# =========================
# CONFIG
# =========================
DATASET_ROOT = Path("/Users/ygthnzdmr/Desktop/project/DataSets/Chess")
TEST_DIR = Path("/Users/ygthnzdmr/Desktop/project/DataSets/Test")
MODEL_PATH = "chess_piece_class.pkl"

UPLOAD_DIR = Path("static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.70

training_progress = {
    "current": 0,
    "total": 0,
    "running": False
}

training_state = {
    "trained": False,
    "run_id" : 0
}
last_accuracy = None

# =========================
# DEVICE
# =========================
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# =========================
# GLOBAL DATALOADER + MODEL
# =========================
dls = ImageDataLoaders.from_folder(
    DATASET_ROOT,
    valid_pct=0.2,
    seed=42,
    item_tfms=Resize(224),
    bs=16,
    device=device
)

learn = cnn_learner(
    dls,
    resnet34,
    metrics=accuracy
)

# =========================
# TRAINING FUNCTION
# =========================
class ProgressCB(Callback):
    def after_epoch(self):
        training_progress["current"] = self.epoch + 1


def train_model_background(epochs: int):
    global training_progress, last_accuracy, learn
    training_state["run_id"] = int(time.time())

    GRAPH_FILES = [
        "static/learning_curve.png",
        "static/confusion_matrix.png",
        "static/roc_curve.png"
    ]
    for f in GRAPH_FILES:
        if os.path.exists(f):
            os.remove(f)

    training_state["trained"] = False
    training_progress["running"] = True
    training_progress["current"] = 0
    training_progress["total"] = epochs

    progress_cb = ProgressCB()

    # ✅ TEK fit
    learn.fit(epochs, cbs=[progress_cb])
    training_progress["current"] = epochs

    # =========================
    # LEARNING CURVE
    # =========================
    train_losses = [float(x) for x in learn.recorder.losses]          # iterasyon bazlı
    values = list(learn.recorder.values)                              # epoch bazlı: (valid_loss, metric...)

    valid_losses = [float(v[0]) for v in values if v and v[0] is not None]  # epoch valid_loss
    accs = []
    # Epoch accuracy (varsa) -> sayfada göstermek için
    last_epoch_acc = accs[-1] if len(accs) > 0 else None
    if len(values) > 0 and len(values[0]) > 1:
        accs = [float(v[1]) for v in values if v and v[1] is not None]      # epoch accuracy

    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss", alpha=0.6)

    if len(valid_losses) > 0:
        x_valid = np.linspace(0, max(len(train_losses)-1, 1), len(valid_losses))
        plt.plot(x_valid, valid_losses, label="Valid Loss", linewidth=3)

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/learning_curve.png")
    plt.close()

    # =========================
    # CONFUSION MATRIX
    # =========================
    preds, targs = learn.get_preds()
    cm = confusion_matrix(targs, preds.argmax(dim=1))

    plt.figure(figsize=(6,6))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=dls.vocab,
        yticklabels=dls.vocab,
        cmap="Blues"
    )
    plt.tight_layout()
    plt.savefig("static/confusion_matrix.png")
    plt.close()

    # =========================
    # ROC CURVE
    # =========================
    y_true = label_binarize(targs, classes=range(len(dls.vocab)))
    y_score = preds.numpy()

    plt.figure()
    for i, cls in enumerate(dls.vocab):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")

    plt.plot([0,1], [0,1], "k--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/roc_curve.png")
    plt.close()

    # =========================
    # ACCURACY
    # =========================
    acc = (preds.argmax(dim=1) == targs).float().mean().item()
    last_accuracy = last_epoch_acc if last_epoch_acc is not None else acc


    training_progress["running"] = False
    training_state["trained"] = True


# =========================
# FLASK
# =========================
app = Flask(__name__)
@app.route("/clear", methods=["POST"])
def clear():
    return redirect(url_for("index"))

@app.route("/progress")
def progress():
    return jsonify({
        "current": training_progress["current"],
        "total": training_progress["total"],
        "running": training_progress["running"],
        "trained": training_state["trained"],
        "run_id": training_state["run_id"]
    })

@app.route("/train", methods=["POST"])
def train():
    epochs = int(request.form.get("epochs", 10))
    training_progress["current"] = 0
    training_progress["total"] = epochs
    training_progress["running"] = True
    training_state["trained"] = False

    thread = threading.Thread(target=train_model_background, args=(epochs,), daemon=True)
    thread.start()
    return ("", 204)  # sadece başlat, cevap verme

@app.route("/test_image/<name>")
def test_image(name):
    return send_file(TEST_DIR / name)

@app.route("/", methods=["GET", "POST"])
def index():
    
    img_path = request.args.get("img")
    result = request.args.get("res")
    conf = request.args.get("conf")
    confidence = float(conf) if conf else None
    note = None
    
    trained = training_state["trained"]

    if request.method == "POST" and not training_progress["running"]:

        if "random_test" in request.form:
            img_name = random.choice(os.listdir(TEST_DIR))
            img = PILImage.create(TEST_DIR / img_name)
            
            pred, idx, probs = learn.predict(img)
            confidence = probs[idx].item()
            result = str(pred) if confidence >= CONF_THRESHOLD else "Emin değilim"

            img_url = url_for("test_image", name=img_name)
            return redirect(url_for("index",
                img=img_url,
                res=result,
                conf=confidence))

        file = request.files.get("image")
        if file and file.filename:
            save_path = UPLOAD_DIR / file.filename
            file.save(save_path)

            img = PILImage.create(save_path)
            pred, idx, probs = learn.predict(img)
            confidence = probs[idx].item()
            result = str(pred) if confidence >= CONF_THRESHOLD else "Emin değilim"

            img_url = url_for("static", filename=f"uploads/{file.filename}")
            return redirect(url_for("index",
                img=img_url,
                res=result,
                conf=confidence))
    graphs = {
        "learning": os.path.exists("static/learning_curve.png"),
        "confusion": os.path.exists("static/confusion_matrix.png"),
        "roc": os.path.exists("static/roc_curve.png")
    }

    if not trained:
        note = "Model henüz eğitilmedi. Test için önce eğitim başlat."
        return render_template(
            "index.html",
            trained = False,
            note=note,
            img_path=None,
            result=None,
            confidence=None,
            accuracy=None,
            graphs=graphs,
            run_id=training_state["run_id"]
        )

    return render_template(
        "index.html",
        trained=True,
        note=note,
        img_path=img_path,
        result=result,
        confidence=confidence,
        accuracy=last_accuracy,
        graphs=graphs,
        run_id=training_state["run_id"]
    )

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
