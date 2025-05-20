import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from preprocess import load_data

# Loading the data
df = load_data("/Users/dml/Downloads/fraud-detection/data/creditcard.csv")

x = df.drop("Class", axis=1)
y = df["Class"]

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

# Handling the imbalanced data
smote = SMOTE(sampling_strategy=0.5, random_state=10)
x_train, y_train = smote.fit_resample(x_train, y_train)

# Training the model
model = RandomForestClassifier(n_estimators=100, random_state=10)
model.fit(x_train, y_train)

# Evaluating the model
y_pred = model.predict(x_test)
print("Model Performance:\n", classification_report(y_test, y_pred))

# Saving the model
with open("/Users/dml/Downloads/fraud-detection/models/model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model save completed!")
