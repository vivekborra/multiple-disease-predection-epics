from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# -------- Load ML Models --------
# Use absolute path based on script location
MODEL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
models = {}
try:
    models = {
        "breast_cancer": joblib.load(os.path.join(MODEL_FOLDER, "cancer.pkl")),
        "diabetes": joblib.load(os.path.join(MODEL_FOLDER, "diabetes.pkl")),
        "heart_disease": joblib.load(os.path.join(MODEL_FOLDER, "heart.pkl")),
        "stroke": joblib.load(os.path.join(MODEL_FOLDER, "stroke.pkl"))
    }
    print("Models loaded:", list(models.keys()))
except Exception as e:
    print("Error loading models:", e)

# -------- Define expected fields for each disease --------
EXPECTED_FIELDS = {
    "diabetes": ["pregnancies","glucose","bloodpressure","skinthickness","insulin","bmi","diabetespedigree","age"],
    "heart_disease": ["age","gender","chestpain","restingBP","serumcholestrol","fastingbloodsugar","restingrelectro","maxheartrate","exerciseangia","oldpeak","slope","noofmajorvessels"],
    "breast_cancer": [
        "age", "race", "marital_status", "t_stage", "n_stage", 
        "sixth_stage", "differentiate", "grade", "a_stage", 
        "tumor_size", "estrogen_status", "progesterone_status",
        "regional_node_examined", "reginol_node_positive", "survival_months"
    ],
    "stroke": ["Chest Pain","Shortness of Breath","Irregular Heartbeat","Fatigue & Weakness","Dizziness","Swelling (Edema)","Pain in Neck/Jaw/Shoulder/Back","Excessive Sweating","Persistent Cough","Nausea/Vomiting","High Blood Pressure","Chest Discomfort (Activity)","Cold Hands/Feet","Snoring/Sleep Apnea","Anxiety/Feeling of Doom","Age","Stroke Risk (%)"]
}

# -------- Label encoding mappings for breast cancer categorical fields --------
BREAST_CANCER_ENCODINGS = {
    "race": {"Black": 0, "Other": 1, "White": 2},
    "marital_status": {"Divorced": 0, "Married": 1, "Separated": 2, "Single": 3, "Widowed": 4},
    "t_stage": {"T1": 0, "T2": 1, "T3": 2, "T4": 3},
    "n_stage": {"N1": 0, "N2": 1, "N3": 2},
    "sixth_stage": {"IIA": 0, "IIB": 1, "IIIA": 2, "IIIB": 3, "IIIC": 4},
    "differentiate": {"Moderately differentiated": 0, "Poorly differentiated": 1, "Undifferentiated": 2, "Well differentiated": 3},
    "grade": {"1": 0, "2": 1, "3": 2, "anaplastic": 3},
    "a_stage": {"Distant": 0, "Regional": 1},
    "estrogen_status": {"Negative": 0, "Positive": 1},
    "progesterone_status": {"Negative": 0, "Positive": 1}
}


def norm(k: str) -> str:
    """Normalize form keys for flexible matching."""
    return k.strip().lower().replace(" ", "_")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check disease selection
        disease = request.form.get("disease")
        if not disease:
            return render_template("result.html", disease="Input Error", result="⚠ No disease selected.")
        
        disease_key = disease.strip().lower()
        if disease_key not in models:
            return render_template("result.html", disease="Error", result="⚠ Model not found. Check model files.")

        # Collect features
        features = []
        missing_fields = []
        for field in EXPECTED_FIELDS.get(disease_key, []):
            value = request.form.get(field)
            if value is None:
                # Try normalized keys
                for k in request.form.keys():
                    if norm(k) == field:
                        value = request.form.get(k)
                        break
            if value is None or value.strip() == "":
                missing_fields.append(field)
            else:
                try:
                    # Special handling for breast cancer categorical fields
                    if disease_key == "breast_cancer" and field in BREAST_CANCER_ENCODINGS:
                        # Encode categorical value to number
                        encoded_value = BREAST_CANCER_ENCODINGS[field].get(value.strip())
                        if encoded_value is None:
                            return render_template(
                                "result.html",
                                disease="Input Error",
                                result=f"⚠ Invalid value '{value}' for field '{field}'. Please select a valid option."
                            )
                        features.append(float(encoded_value))
                    else:
                        # Regular numeric field
                        features.append(float(value.strip()))
                except ValueError:
                    return render_template(
                        "result.html",
                        disease="Input Error",
                        result=f"⚠ Field '{field}' must be numeric. Got '{value}'."
                    )

        if missing_fields:
            return render_template(
                "result.html",
                disease="Input Error",
                result=f"⚠ Missing fields: {', '.join(missing_fields)}. Check your form."
            )

        if len(features) == 0:
            return render_template("result.html", disease="Input Error", result="⚠ No input features found.")

        X = np.array(features).reshape(1, -1)

        # Check model input size
        model = models[disease_key]
        if hasattr(model, "n_features_in_") and X.shape[1] != model.n_features_in_:
            return render_template(
                "result.html",
                disease="Input Error",
                result=f"⚠ Number of inputs ({X.shape[1]}) does not match model expectation ({model.n_features_in_})."
            )

        # Make prediction
        pred = model.predict(X)
        prediction_value = pred[0]
        if isinstance(prediction_value, (int, np.integer)):
            result_text = "🛑 Positive — High Risk" if int(prediction_value) == 1 else "✅ Negative — Low Risk"
        else:
            result_text = f"Prediction: {prediction_value}"

        return render_template("result.html", disease=disease.replace("_", " ").title(), result=result_text)

    except Exception as e:
        return render_template("result.html", disease="System Error", result=f"⚠ Unexpected error: {e}")

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # use Render's PORT if set
    app.run(host="0.0.0.0", port=port, debug=False)
