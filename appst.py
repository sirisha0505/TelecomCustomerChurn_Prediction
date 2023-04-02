#Streamlit File

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# import libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


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
data = pd.read_csv('train.csv')

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


# Load the saved model
model = joblib.load("rf_model.joblib")


# Define a function to perform label encoding on categorical columns
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

# Define the Streamlit app
def app():
    st.set_page_config(page_title="Telecom Churn Prediction", page_icon=":shamrock:", layout="wide")
    st.title('                             Telecom Churn Prediction                           ')
    st.image("https://www.livechatinc.com/wp-content/uploads/2016/04/customer-churn.jpg",width=800,caption='Fish In - Fish Out ')
    st.write('#### Please enter the customer details to predict whether the customer is likely to churn or not.')
   
    # Define the input form
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female'])
        SeniorCitizen = st.selectbox('Senior citizen',[0,1])
        partner = st.selectbox('Partner', ['Yes', 'No'])
        dependents = st.selectbox('Dependents', ['Yes', 'No'])
        phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
        multiple_lines = st.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])
        internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
        online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
        device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
        
    with col2:
        tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
        streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
        streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
        contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
        payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        tenure = st.number_input('Tenure (months)', min_value=1, max_value=100, value=1)
        monthly_charges = st.number_input('Monthly Charges ($)', min_value=1.0, max_value=1000.0, value=50.0)
        total_charges = st.number_input('Total Charges ($)', min_value=1.0, max_value=100000.0, value=50.0)

    # Create a dictionary with the input values
    input_dict = {'gender': gender, 'SeniorCitizen': 0, 'Partner': partner, 'Dependents': dependents, 
                  'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': multiple_lines, 
                  'InternetService': internet_service, 'OnlineSecurity': online_security, 'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
              'TechSupport': tech_support, 'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
              'Contract': contract, 'PaperlessBilling': paperless_billing, 'PaymentMethod': payment_method,
              'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges}

    if st.button('Predict 👈'):
        # Convert the input dictionary to a Pandas DataFrame
        new_data = pd.DataFrame.from_dict([input_dict])

        # Perform label encoding on categorical columns
        new_data = label_encode(new_data)

        # Make predictions using the loaded RandomForest  model
        predictions = model.predict_proba(new_data)

        # Display the prediction result
        if predictions.shape[1] == 2:
            churn_probability = predictions[0][1]
            if churn_probability >= 0.5:
                st.warning('The customer is likely to churn with a probability of {}%.'.format(round(churn_probability*100, 2)))
            else:
                st.success('The customer is unlikely to churn with a probability of {}%.'.format(round((1-churn_probability)*100, 2)))
        else:
            st.error('Error: Invalid model, please check the model file.')



if __name__ == '__main__':
    app()
