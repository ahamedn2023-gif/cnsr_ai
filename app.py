import os

# Tell TensorFlow not to look for a GPU on Render
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import uuid
import sqlite3
import numpy as np

from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, session

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
app = Flask(__name__)
app.secret_key = "change-this-secret-key"

# -----------------------------
# Configuration
# -----------------------------
UPLOAD_FOLDER = "static/uploads"
DATABASE = "database.db"


app.config["UPLOAD_FOLDER"] = os.path.join(
    app.static_folder,
    "uploads"
)

os.makedirs(
    app.config["UPLOAD_FOLDER"],
    exist_ok=True
)



# -----------------------------
# Database
# -----------------------------
def init_db():
    conn = sqlite3.connect(DATABASE)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            anonymous_id TEXT NOT NULL,
            image_name TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


init_db()


# -----------------------------
# Load AI model
# -----------------------------
model = load_model("model.h5")


# -----------------------------
# Anonymous Login
# -----------------------------
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        # Generate anonymous ID
        anonymous_id = uuid.uuid4().hex[:8].upper()

        session["anonymous_id"] = anonymous_id

        return redirect(url_for("home"))

    return render_template("login.html")


# -----------------------------
# Home
# -----------------------------
@app.route("/")
def home():

    if "anonymous_id" not in session:
        return redirect(url_for("login"))

    return render_template(
        "index.html",
        anonymous_id=session["anonymous_id"]
    )


# -----------------------------
# Prediction
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "anonymous_id" not in session:
        return redirect(url_for("login"))

    file = request.files.get("image")

    if not file or file.filename == "":
        return redirect(url_for("home"))

    # Give uploaded image a unique filename
    extension = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4().hex}{extension}"

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    # -----------------------------
    # Prepare image
    # -----------------------------
    img = image.load_img(
        filepath,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    # -----------------------------
    # AI prediction
    # -----------------------------
    prediction_value = model.predict(img_array, verbose=0)

    # IMPORTANT:
    # This assumes your model outputs:
    # 0 = benign
    # 1 = malignant

    probability = float(prediction_value[0][0])

    if probability >= 0.5:
        prediction = "Malignant"
        confidence = probability * 100
    else:
        prediction = "Benign"
        confidence = (1 - probability) * 100

    # -----------------------------
    # Save to database
    # -----------------------------
    anonymous_id = session["anonymous_id"]

    conn = sqlite3.connect(DATABASE)

    conn.execute("""
        INSERT INTO predictions
        (anonymous_id, image_name, prediction, confidence, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        anonymous_id,
        filename,
        prediction,
        confidence,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

    # -----------------------------
    # Show result
    # -----------------------------
    return render_template(
     "result.html",
      prediction=prediction,
     confidence=round(confidence, 2),
     image_path=url_for("static", filename=f"uploads/{filename}"),
     anonymous_id=anonymous_id
     )


# -----------------------------
# Prediction History
# -----------------------------
@app.route("/history")
def history():

    if "anonymous_id" not in session:
        return redirect(url_for("login"))

    anonymous_id = session["anonymous_id"]

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    records = conn.execute("""
        SELECT *
        FROM predictions
        WHERE anonymous_id = ?
        ORDER BY id DESC
    """, (anonymous_id,)).fetchall()

    conn.close()

    return render_template(
        "history.html",
        records=records,
        anonymous_id=anonymous_id
    )


# -----------------------------
# Logout
# -----------------------------
@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))


# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)