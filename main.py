# Imports

from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import Satellite_model

# Read the data
df = pd.read_json('shipsnet.json')

# Normalize and reshape the image data
df["normalized_data"] = df["data"].apply(lambda x: (np.array(x) / 255).reshape(80, 80, 3))

# Define X and y
X = df['normalized_data']
y = df['labels']

# Split into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Transform the training and testing data into arrays
X_train = np.array([data for data in X_train])
X_test = np.array([data for data in X_test])
Y_train = np.array([data for data in Y_train])
Y_test = np.array([data for data in Y_test])

# Inherit model
satellitemodel = Satellite_model.SatelliteModel(X_train, Y_train, X_test, Y_test, 20, 200, 20)

# Make predictions
model, predictions = satellitemodel.make_prediction()

# Evaluate model
print(classification_report(Y_test, predictions.round()))
print(confusion_matrix(Y_test, predictions.round()))

model.save("satellitemodel.h5")


