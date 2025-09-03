from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

# tell Flask to look in "frontend" instead of "templates"
app = Flask(__name__, template_folder="frontend")
CORS(app)

# Load model + encoders
model = joblib.load("co2_model.pkl")
le_dict = joblib.load("encoders.pkl")

def safe_encode(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        le_classes = list(le.classes_)
        le_classes.append(value)
        le.classes_ = np.array(le_classes)
        return le.transform([value])[0]

@app.route("/")
def home():
    return render_template("index.html")   # loads from frontend/

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    encoded = []
    for col in ["Make", "Model", "Vehicle Class", "Transmission", "Fuel Type"]:
        encoded.append(safe_encode(le_dict[col], data[col]))

    encoded.extend([float(data["Engine Size(L)"]), int(data["Cylinders"])])

    X_new = pd.DataFrame([encoded], columns=model.feature_names_in_)
    prediction = model.predict(X_new)[0]

    return jsonify({"prediction": round(float(prediction), 2)})

if __name__ == "__main__":
    # Render requires host=0.0.0.0 and port from env
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
