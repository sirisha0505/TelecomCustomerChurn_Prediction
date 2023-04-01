# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def label_encode(data):
    # Perform label encoding on categorical columns
    le = LabelEncoder()
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                        'MultipleLines', 'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract',
                        'PaperlessBilling', 'PaymentMethod']
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    
    return data
# load data
data = pd.read_csv(r'C:\Users\gnssi\Desktop\capstone\hackthon\train.csv')

# data preprocessing
data = data.drop(['customerID'], axis=1) # drop unnecessary column
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.dropna() # drop missing values

# encode
data=label_encode(data)

# create feature matrix and target vector
X = data.drop(['Churn'], axis=1)
y = data['Churn']

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# create a Random Forest Classifier model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# fit the model with training data
rf.fit(X_train, y_train)

# make predictions on testing data
y_pred = rf.predict(X_test)

# evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# save the model
import joblib
joblib.dump(rf, 'rf_model.joblib')
