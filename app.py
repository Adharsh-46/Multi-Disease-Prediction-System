import os
import pickle
import sqlite3
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, session, redirect, url_for, g
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash
import joblib

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")

# ---------------------------------------------------------
# 1. ROBUST PATH CONFIGURATION
# ---------------------------------------------------------
# This finds the folder where app.py lives automatically
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
USER_DB = BASE_DIR / "users.db"


# ---------------------------------------------------------
# 1b. AUTH / USER STORE (SQLite)
# ---------------------------------------------------------
def init_db():
    """Create users table if it doesn't exist."""
    with sqlite3.connect(USER_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
            """
        )
        conn.commit()


def get_user(username):
    """Return user row or None."""
    with sqlite3.connect(USER_DB) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
        return cur.fetchone()


def add_user(username, password):
    """Insert new user with hashed password."""
    password_hash = generate_password_hash(password)
    with sqlite3.connect(USER_DB) as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash),
        )
        conn.commit()


@app.before_request
def load_logged_in_user():
    """Expose current user to templates via g.user."""
    g.user = session.get("username")


# Initialize the auth database on startup
init_db()

def load_pkl(filename):
    path = MODELS_DIR / filename
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return None

def load_h5(filename):
    path = MODELS_DIR / filename
    try:
        # Path must be converted to string for Keras
        model = tf.keras.models.load_model(str(path))
        print(f"✅ {filename} loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return None

# ---------------------------------------------------------
# 2. LOAD MODELS (LOADED ONCE AT STARTUP)
# ---------------------------------------------------------
diabetes_model = load_pkl('diabetes_model.sav')
cancer_model = load_pkl('cancer.sav')
heart_model = load_pkl('heart_disease_model.sav')
kidney_model = load_pkl('kidney.sav')
liver_model = load_pkl('liver_model.sav')

malaria_model = load_h5("malaria.h5")
pneumonia_model = load_h5("pneumonia_model.h5")

# ---------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------
def predict_logic(values, dic):
    """Handles logic for Sklearn/Pickle models based on input length."""
    # Map input count to model
    model_map = {8: diabetes_model, 22: cancer_model, 13: heart_model, 24: kidney_model, 10: liver_model}
    model = model_map.get(len(values))
    
    if model:
        return model.predict(np.asarray(values).reshape(1, -1))[0]
    return None

# ---------------------------------------------------------
# 4. ROUTES
# ---------------------------------------------------------
@app.route("/")
def home():
    if not g.user:
        return redirect(url_for("login"))
    return render_template("home.html")

@app.route("/diabetes")
def diabetesPage(): return render_template("diabetes.html")

@app.route("/cancer")
def cancerPage(): return render_template("breast_cancer.html")

@app.route("/heart")
def heartPage(): return render_template("heart.html")

@app.route("/kidney")
def kidneyPage(): return render_template("kidney.html")

@app.route("/liver")
def liverPage(): return render_template("liver.html")

@app.route("/malaria")
def malariaPage(): return render_template("malaria.html")

@app.route("/pneumonia")
def pneumoniaPage(): return render_template("pneumonia.html")

# ---------------------------------
# Auth Routes
# ---------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    message = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            message = "Username and password are required."
        elif len(password) < 6:
            message = "Password must be at least 6 characters."
        elif get_user(username):
            message = "Username already taken."
        else:
            try:
                add_user(username, password)
                session["username"] = username
                return redirect(url_for("home"))
            except Exception as e:
                message = f"Could not create user: {e}"
    return render_template("register.html", message=message)


@app.route("/login", methods=["GET", "POST"])
def login():
    message = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = get_user(username) if username else None
        if user and check_password_hash(user["password_hash"], password):
            session["username"] = username
            return redirect(url_for("home"))
        message = "Invalid username or password."
    return render_template("login.html", message=message)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# ---------------------------------
# Standard Disease Prediction
# ---------------------------------
@app.route("/predict", methods=['POST'])
def predictPage():
    try:
        to_predict_dict = {k: float(v) for k, v in request.form.items() if v != ""}
        if len(to_predict_dict) != len(request.form):
            return render_template("predict.html", message="Please fill all fields")

        to_predict_list = list(to_predict_dict.values())
        pred = predict_logic(to_predict_list, to_predict_dict)
        return render_template("predict.html", pred=pred)
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return render_template("predict.html", message="Invalid input data")

# ---------------------------------
# Malaria Prediction
# ---------------------------------
@app.route("/malariapredict", methods=['POST'])
def malariapredictPage():

    if malaria_model is None:
        return render_template("malaria.html", message="Malaria model not available")

    try:
        file = request.files.get('image')

        if not file or file.filename == "":
            return render_template("malaria.html", message="Please select an image")

        img = Image.open(file).convert("RGB").resize((128, 128))
        img = np.array(img) / 255.0
        img = img.reshape(1, 128, 128, 3)

        prediction = malaria_model.predict(img, verbose=0)
        prediction = np.asarray(prediction)

        print("RAW MALARIA:", prediction)

        if prediction.ndim == 2 and prediction.shape[1] == 2:
            confidence = float(prediction[0][1])
        elif prediction.ndim == 2 and prediction.shape[1] == 1:
            confidence = float(prediction[0][0])
        elif prediction.ndim == 1:
            confidence = float(prediction[0])
        else:
            raise ValueError(f"Unexpected malaria prediction shape: {prediction.shape}")

        pred = 1 if confidence >= 0.5 else 0
        confidence_percent = round(confidence * 100, 2)

        print(f"MALARIA -> Conf: {confidence_percent}% | Pred: {pred}")

        return render_template(
            "predict.html",      # ✅ COMMON PAGE
            pred=int(pred),
            confidence=confidence_percent
        )

    except Exception as e:
        app.logger.error(f"Malaria error: {e}")
        return render_template("malaria.html", message="Error processing image")
# ---------------------------------
# Pneumonia Prediction 
# ---------------------------------
@app.route("/pneumoniapredict", methods=['POST'])
def pneumoniapredictPage():

    if pneumonia_model is None:
        return render_template("pneumonia.html", message="Pneumonia model not available")

    try:
        file = request.files.get('image')

        if not file or file.filename == "":
            return render_template("pneumonia.html", message="Please upload an image")

        # ✅ Correct preprocessing (match training)
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))

        img = np.array(img) / 255.0
        img = img.reshape(1, 224, 224, 3)

        # Prediction
        prediction = pneumonia_model.predict(img, verbose=0)
        prediction = np.asarray(prediction)

        print("RAW PNEUMONIA:", prediction)

        # ✅ Handle all output shapes
        if prediction.ndim == 2 and prediction.shape[1] == 2:
            confidence = float(prediction[0][1])   # softmax
        elif prediction.ndim == 2 and prediction.shape[1] == 1:
            confidence = float(prediction[0][0])   # sigmoid
        elif prediction.ndim == 1:
            confidence = float(prediction[0])
        else:
            raise ValueError(f"Unexpected prediction shape: {prediction.shape}")

        # ✅ Slightly lower threshold
        pred = 1 if confidence >= 0.3 else 0

        confidence_percent = round(confidence * 100, 2)

        print(f"PNEUMONIA -> Conf: {confidence_percent}% | Pred: {pred}")

        return render_template(
            "predict.html",          # ✅ USE COMMON PAGE
            pred=int(pred),          # ✅ ALWAYS numeric
            confidence=confidence_percent
        )

    except Exception as e:
        app.logger.error(f"Pneumonia error: {e}")
        return render_template("pneumonia.html", message=f"Error: {str(e)}")
if __name__ == "__main__":
    app.run(debug=True)