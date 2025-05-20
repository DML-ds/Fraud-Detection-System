import pickle
import numpy as np

# loading the trained model
with open("/Users/dml/Downloads/fraud-detection/models/model.pkl", "rb") as file:
    model = pickle.load(file)

# testing
transaction = np.array([0.1, -0.5, 0.3, 1.2, -0.9] + [0] * 24). reshape(1, -1)

# could it be fraud?
prediction = model.predict(transaction)
print("Fraud Detected" if prediction [0] == 1 else "Not Fraud")
