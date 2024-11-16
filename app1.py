# coding: utf-8

import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")

# Load the data to align column structure and train data
df_1 = pd.read_csv("first_telc.csv")

# Load the pre-trained models
model_dt = pickle.load(open("model_dt.sav", "rb"))
model_rf = pickle.load(open("model_rf.sav", "rb"))

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    # Collect input data from the form
    input_data = [
        request.form['query1'],  # SeniorCitizen
        request.form['query2'],  # MonthlyCharges
        request.form['query3'],  # TotalCharges
        request.form['query4'],  # gender
        request.form['query5'],  # Partner
        request.form['query6'],  # Dependents
        request.form['query7'],  # PhoneService
        request.form['query8'],  # MultipleLines
        request.form['query9'],  # InternetService
        request.form['query10'], # OnlineSecurity
        request.form['query11'], # OnlineBackup
        request.form['query12'], # DeviceProtection
        request.form['query13'], # TechSupport
        request.form['query14'], # StreamingTV
        request.form['query15'], # StreamingMovies
        request.form['query16'], # Contract
        request.form['query17'], # PaperlessBilling
        request.form['query18'], # PaymentMethod
        request.form['query19']  # tenure
    ]

    # Create a DataFrame for the input data
    data_columns = [
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'tenure'
    ]
    new_df = pd.DataFrame([input_data], columns=data_columns)

    # Concatenate with df_1 to maintain structure for dummy variables
    df_2 = pd.concat([df_1, new_df], ignore_index=True)

    # Add tenure_group column
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)

    # Drop unnecessary columns
    df_2.drop(columns=['tenure'], axis=1, inplace=True)

    # One-hot encode the categorical variables
    new_df_dummies = pd.get_dummies(df_2[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                           'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                           'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']])

    # Check for duplicate columns and handle them
    print("Checking for duplicate columns...")

    if new_df_dummies.columns.duplicated().any():
        print("Duplicate columns found in new_df_dummies. Removing duplicates.")
        new_df_dummies = new_df_dummies.loc[:, ~new_df_dummies.columns.duplicated()]

    expected_columns = model_dt.feature_names_in_  # Assuming both models use the same columns

    if len(expected_columns) != len(set(expected_columns)):
        print("Duplicate columns found in expected_columns. Removing duplicates.")
        expected_columns = list(dict.fromkeys(expected_columns))

    # Align with model input structure
    new_df_dummies = new_df_dummies.reindex(columns=expected_columns, fill_value=0)

    # Make predictions using both models
    dt_prediction = model_dt.predict(new_df_dummies.tail(1))
    dt_prob = model_dt.predict_proba(new_df_dummies.tail(1))[:, 1]
    
    rf_prediction = model_rf.predict(new_df_dummies.tail(1))
    rf_prob = model_rf.predict_proba(new_df_dummies.tail(1))[:, 1]

    # Interpret the Decision Tree model's result
    if dt_prediction == 1:
        dt_output = "Decision Tree: This customer is likely to churn."
    else:
        dt_output = "Decision Tree: This customer is likely to continue."

    dt_confidence = "Confidence: {:.2f}%".format(dt_prob[0] * 100)

    # Interpret the Random Forest model's result
    if rf_prediction == 1:
        rf_output = "Random Forest: This customer is likely to churn."
    else:
        rf_output = "Random Forest: This customer is likely to continue."

    rf_confidence = "Confidence: {:.2f}%".format(rf_prob[0] * 100)

    # Render the results on the template
    return render_template(
        'home.html',
        output1=dt_output, confidence1=dt_confidence,
        output2=rf_output, confidence2=rf_confidence,
        query1=request.form['query1'],
        query2=request.form['query2'],
        query3=request.form['query3'],
        query4=request.form['query4'],
        query5=request.form['query5'],
        query6=request.form['query6'],
        query7=request.form['query7'],
        query8=request.form['query8'],
        query9=request.form['query9'],
        query10=request.form['query10'],
        query11=request.form['query11'],
        query12=request.form['query12'],
        query13=request.form['query13'],
        query14=request.form['query14'],
        query15=request.form['query15'],
        query16=request.form['query16'],
        query17=request.form['query17'],
        query18=request.form['query18'],
        query19=request.form['query19']
    )

# Run the application
if __name__ == "__main__":
    app.run(debug=True)
