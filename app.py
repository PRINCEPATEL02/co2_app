import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# --- Robust paths so it works on Render too ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "co2_model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")

# Load model + encoders at startup
model = joblib.load(MODEL_PATH)
le_dict = joblib.load(ENCODERS_PATH)

def safe_encode(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    # Add unseen category dynamically
    le_classes = list(le.classes_)
    le_classes.append(value)
    le.classes_ = np.array(le_classes)
    return le.transform([value])[0]

@app.route("/", methods=["GET"])
def home():
    # Serves templates/index.html
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Encode categoricals
    encoded = []
    for col in ["Make", "Model", "Vehicle Class", "Transmission", "Fuel Type"]:
        encoded.append(safe_encode(le_dict[col], data[col]))

    # Add numeric features
    encoded.extend([float(data["Engine Size(L)"]), int(data["Cylinders"])])

    # Match training feature order
    X_new = pd.DataFrame([encoded], columns=model.feature_names_in_)
    yhat = model.predict(X_new)[0]
    return jsonify({"prediction": round(float(yhat), 2)})

if __name__ == "__main__":
    # Local dev only; Render will use gunicorn to serve app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
