import os
import pandas as pd
from flask import Flask, request, render_template
import pickle
from tensorflow.keras.models import load_model
import numpy as np
from flask import redirect, url_for

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask("__name__")

# Load the data and models
df_1 = pd.read_csv("first_telc.csv")
model_dt = pickle.load(open("model_dt.sav", "rb"))
model_rf = pickle.load(open("model_rf.sav", "rb"))
model_ann = load_model("model_ann.keras")

# @app.route("/")
# def loadPage():
#     return render_template('home1.html', query="", output1=None, output2=None, output3=None)

@app.route("/")
def loadPage():
    # Clear data and outputs on page load
    return render_template('home1.html', query1=None, query2=None, query3=None, query4=None, 
                           query5=None, query6=None, query7=None, query8=None, query9=None,
                           query10=None, query11=None, query12=None, query13=None, query14=None,
                           query15=None, query16=None, query17=None, query18=None, query19=None,
                           output1=None, confidence1=None, output2=None, confidence2=None,
                           output3=None, confidence3=None)


@app.route("/", methods=['POST'])
def predict():
    input_data = [
        request.form['query1'], request.form['query2'], request.form['query3'],
        request.form['query4'], request.form['query5'], request.form['query6'],
        request.form['query7'], request.form['query8'], request.form['query9'],
        request.form['query10'], request.form['query11'], request.form['query12'],
        request.form['query13'], request.form['query14'], request.form['query15'],
        request.form['query16'], request.form['query17'], request.form['query18'],
        request.form['query19']
    ]

    data_columns = [
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'tenure'
    ]
    new_df = pd.DataFrame([input_data], columns=data_columns)
    df_2 = pd.concat([df_1, new_df], ignore_index=True)

    # Creating tenure_group
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
    df_2.drop(columns=['tenure'], axis=1, inplace=True)

    # One-hot encode and handle duplicate columns
    new_df_dummies = pd.get_dummies(df_2[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                          'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                          'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                          'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']])

    new_df_dummies = new_df_dummies.loc[:, ~new_df_dummies.columns.duplicated()]
    expected_columns = model_dt.feature_names_in_
    new_df_dummies = new_df_dummies.reindex(columns=expected_columns, fill_value=0)

    # Predict with Decision Tree and Random Forest
    dt_prediction = model_dt.predict(new_df_dummies.tail(1))
    dt_prob = model_dt.predict_proba(new_df_dummies.tail(1))[:, 1]
    rf_prediction = model_rf.predict(new_df_dummies.tail(1))
    rf_prob = model_rf.predict_proba(new_df_dummies.tail(1))[:, 1]

    # Predict with ANN model (convert to float32 numpy array)
    ann_input = new_df_dummies.tail(1).to_numpy().astype(np.float32)
    ann_prob = model_ann.predict(ann_input)[0][0]  # Assuming binary output from ANN model

    # Interpret results
    # dt_output = "Decision Tree: This customer is likely to churn." if dt_prediction == 1 else "Decision Tree: This customer is likely to continue."
    # dt_confidence = f"Confidence: {dt_prob[0] * 100:.2f}%"
    if dt_prob[0] > 0.5:
        dt_output = "Decision Tree: This customer is likely to churn."
        dt_confidence = f"Confidence: {dt_prob[0] * 100:.2f}%"
    else:
        dt_output = "Decision Tree: This customer is likely to continue."
        dt_confidence = f"Confidence: {(1 - dt_prob[0]) * 100:.2f}%"

    # rf_output = "Random Forest: This customer is likely to churn." if rf_prediction == 1 else "Random Forest: This customer is likely to continue."
    # rf_confidence = f"Confidence: {rf_prob[0] * 100:.2f}%"
    if rf_prob[0] > 0.5:
        rf_output = "Random Forest: This customer is likely to churn."
        rf_confidence = f"Confidence: {rf_prob[0] * 100:.2f}%"
    else:
        rf_output = "Random Forest: This customer is likely to continue."
        rf_confidence = f"Confidence: {(1 - rf_prob[0]) * 100:.2f}%"

    # ann_output = "ANN: This customer is likely to churn." if ann_prob > 0.5 else "ANN: This customer is likely to continue."
    # ann_confidence = f"Confidence: {ann_prob * 100:.2f}%"
    if ann_prob > 0.5:
        ann_output = "ANN: This customer is likely to churn."
        ann_confidence = f"Confidence: {ann_prob * 100:.2f}%"
    else:
        ann_output = "ANN: This customer is likely to continue."
        ann_confidence = f"Confidence: {(1 - ann_prob) * 100:.2f}%"

    return render_template(
        'home1.html',
        output1=dt_output, confidence1=dt_confidence,
        output2=rf_output, confidence2=rf_confidence,
        output3=ann_output, confidence3=ann_confidence,
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
