import os
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__, template_folder="templates")

# Define paths to model files (relative to the current script)
model_path = os.path.join(os.path.dirname(__file__), "MODEL", "spam_classifier.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "MODEL", "tfidf_vectorizer.pkl")

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("✅ Model and Vectorizer loaded successfully")
except Exception as e:
    print(f"❌ Error loading model/vectorizer: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form['message']
        transformed_data = vectorizer.transform([data])
        prediction = model.predict(transformed_data)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
