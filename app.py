# building an API for real-time detection
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# loading the trained model
with open("/Users/dml/Downloads/fraud-detection/models/model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/predict", methods = ["POST"])
def predict():
    data = request.get_json()
    transaction_features = np.array(data["features"]).reshape(1, -1)

    prediction = model.predict(transaction_features)
    return jsonify({"fraudulent": bool(prediction[0])})

if __name__ == "__main__":
    app.run(debug = True)

