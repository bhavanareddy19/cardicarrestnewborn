import numpy as np
import pandas as pd
from sklearn import *
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Read the dataset into a dataframe
df = pd.read_csv('infantset.csv')

# Map categorical variables to numerical values

df["BirthWeight"] = df["BirthWeight"].map({'WeightTooLow':3 ,'LowWeight':2,'NormalWeight':1})


#Mapping of Family History
df["FamilyHistory"] = df["FamilyHistory"].map({'AboveTwoCases':3 ,'ZeroToTwoCases':2,'NoCases':1})


#Mapping of Preterm Birth
df["PretermBirth"] = df["PretermBirth"].map({'4orMoreWeeksEarlier':3 ,'2To4weeksEarlier':2,'NotaPreTerm':1})


#Mapping of Heart Rate
df["HeartRate"] = df["HeartRate"].map({'RapidHeartRate':3 ,'HighHeartRate':2,'NormalHeartRate':1})


#Mapping of Breathing Difficulty
df["BreathingDifficulty"] = df["BreathingDifficulty"].map({'HighBreathingDifficulty':3 ,'BreathingDifficulty':2,'NoBreathingDifficulty':1})


#Mapping of Skin Tinge
df["SkinTinge"] = df["SkinTinge"].map({'Bluish':3 ,'LightBluish':2,'NotBluish':1})


#Mapping of Responsiveness
df["Responsiveness"] = df["Responsiveness"].map({'UnResponsive':3 ,'SemiResponsive':2,'Responsive':1})


#Mapping of Movement
df["Movement"] = df["Movement"].map({'Diminished':3 ,'Decreased':2,'NormalMovement':1})


#Mapping of Delivery Type
df["DeliveryType"] = df["DeliveryType"].map({'C_Section':3 ,'DifficultDelivery':2,'NormalDelivery':1})


#Mapping of Mothers BP History
df["MothersBPHistory"] = df["MothersBPHistory"].map({'VeryHighBP':3 ,'HighBP':2,'BPInRange':1})


#Mapping of Cardiac Arrest Chance
df["CardiacArrestChance"] = df["CardiacArrestChance"].map({'High':2 ,'Medium':1,'Low':0})


# Convert dataframe to numpy array
data = df[["BirthWeight", "FamilyHistory", "PretermBirth", "HeartRate", "BreathingDifficulty", "SkinTinge", "Responsiveness", "Movement", "DeliveryType", "MothersBPHistory", "CardiacArrestChance"]].to_numpy()

# Split data into inputs and outputs
inputs = data[:, :-1]
outputs = data[:, -1]

# Split data into training and testing sets
training_inputs = inputs[:1000]
training_outputs = outputs[:1000]
testing_inputs = inputs[1000:]
testing_outputs = outputs[1000:]

# Standardize input features
scaler = StandardScaler()
training_inputs_scaled = scaler.fit_transform(training_inputs)
testing_inputs_scaled = scaler.transform(testing_inputs)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_samples': [0.5, 1.0],
    # Add other hyperparameters to tune
}

grid_search = GridSearchCV(BaggingClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(training_inputs_scaled, training_outputs)

best_classifier = grid_search.best_estimator_

# Predictions on testing data
predictions = best_classifier.predict(testing_inputs_scaled)

# Calculate accuracy
accuracy = accuracy_score(testing_outputs, predictions)
print("The accuracy of Bagging Classifier on testing data is: {:.2f}%".format(accuracy * 100))

# Calculate precision, recall, and F1-score
precision = precision_score(testing_outputs, predictions, average='macro')
recall = recall_score(testing_outputs, predictions, average='macro')
f1 = f1_score(testing_outputs, predictions, average='macro')