import io
import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model
MODEL_PATH = "EDDITH.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model file not found! Place EDDITH.h5 in the same directory.")

model = load_model(MODEL_PATH)
CLASS_NAMES = ["Cataract", "Glaucoma", "Diabetic Retinopathy", "Normal"]

# Disease information
disease_info = {
    "Cataract": {
        "symptoms": "Blurry vision, faded colors, sensitivity to light",
        "cure": "Surgical lens replacement",
        "medication": "Prescription glasses (early), post-op eye drops"
    },
    "Glaucoma": {
        "symptoms": "Eye pain, blurred vision, halos around lights",
        "cure": "Surgery or laser therapy to relieve pressure",
        "medication": "Eye drops, oral medications"
    },
    "Diabetic Retinopathy": {
        "symptoms": "Spots or dark strings (floaters), blurred vision",
        "cure": "Laser treatment, vitrectomy",
        "medication": "Anti-VEGF injections, diabetes management drugs"
    },
    "Normal": {
        "symptoms": "None",
        "cure": "Not required",
        "medication": "Not required"
    }
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = tf.keras.preprocessing.image.load_img(io.BytesIO(file.read()), target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction))
    confidence_score = float(np.max(prediction))
    class_name = CLASS_NAMES[predicted_class]

    details = disease_info.get(class_name, {})

    return jsonify({
        "prediction": predicted_class,
        "class_name": class_name,
        "confidence": round(confidence_score, 4),
        "symptoms": details.get("symptoms", "N/A"),
        "cure": details.get("cure", "N/A"),
        "medication": details.get("medication", "N/A")
    })

@app.route('/diagnose', methods=['POST'])
def diagnose():
    data = request.get_json()
    symptoms = data.get('symptoms', [])

    #  logic (replace with ML model or rules)
    if 'eye_pain' in symptoms and 'redness' in symptoms:
        disease = "Conjunctivitis"
        cure = "Cold compress, avoid irritants"
        medication = "Lubricant eye drops, antihistamines"
    elif 'blurred_vision' in symptoms and 'floaters' in symptoms:
        disease = "Diabetic Retinopathy"
        cure = "Control blood sugar, regular eye exams"
        medication = "Anti-VEGF injections, laser surgery"
    else:
        disease = "Unknown"
        cure = "Consult an eye specialist"
        medication = "N/A"

    return jsonify({
        "predicted_disease": disease,
        "cure": cure,
        "medication": medication
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
