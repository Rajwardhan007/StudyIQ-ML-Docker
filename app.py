
from flask import Flask, request, jsonify, render_template
import joblib
import json
import pandas as pd

app = Flask(__name__)

APP_VERSION = "1.1.0"

# Load trained model
model = joblib.load("models/student_model.pkl")

# Load default student profile
with open("models/default_student.json", "r") as file:
    default_student = json.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Please provide student details"}), 400

        # Validate required inputs
        required_fields = [
            "studytime", "failures", "absences", "G1", "G2"
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({
                    "error": f"Missing input: {field}"
                }), 400

        # Convert input values
        studytime = int(data["studytime"])
        failures = int(data["failures"])
        absences = int(data["absences"])
        G1 = float(data["G1"])
        G2 = float(data["G2"])

        # Validate input ranges
        if studytime not in [1, 2, 3, 4]:
            return jsonify({"error": "Study time must be between 1 and 4"}), 400

        if failures not in range(5):
            return jsonify({"error": "Failures must be between 0 and 4"}), 400

        if not 0 <= absences <= 93:
            return jsonify({"error": "Absences must be between 0 and 93"}), 400

        if not 0 <= G1 <= 20 or not 0 <= G2 <= 20:
            return jsonify({"error": "Previous grades must be between 0 and 20"}), 400

        # Start with default profile
        student = default_student.copy()

        # Replace selected features
        student["studytime"] = studytime
        student["failures"] = failures
        student["absences"] = absences
        student["G1"] = G1
        student["G2"] = G2

        # Generate prediction
        input_data = pd.DataFrame([student])
        predicted_grade = float(model.predict(input_data)[0])

        # Grade is already on a 0-20 scale
        predicted_grade = max(0, min(20, predicted_grade))
        percentage = (predicted_grade / 20) * 100

        if predicted_grade >= 16:
            category = "Excellent"
        elif predicted_grade >= 14:
            category = "Very Good"
        elif predicted_grade >= 10:
            category = "Satisfactory"
        else:
            category = "Needs Improvement"

        return jsonify({
            "predicted_grade": round(predicted_grade, 2),
            "out_of": 20,
            "percentage": round(percentage, 2),
            "performance": category,
            "inputs": {
                "studytime": studytime,
                "failures": failures,
                "absences": absences,
                "G1": G1,
                "G2": G2
            }
        })

    except (ValueError, TypeError):
        return jsonify({"error": "Please enter valid numerical values"}), 400

    except Exception as error:
        print("Prediction error:", error)
        return jsonify({"error": "Prediction failed"}), 500



@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "application": "StudyIQ",
        "version": APP_VERSION
    })


@app.route("/version", methods=["GET"])
def version():
    return jsonify({
        "application": "StudyIQ",
        "version": APP_VERSION,
        "model": "Random Forest Regressor"
    })


if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=5000, debug=False)
