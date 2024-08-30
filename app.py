from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the files are present in the request
        if "lead_data" not in request.files or "conversion_data" not in request.files:
            return "Error: Please upload both datasets."

        # Read the input files from the form
        lead_data = pd.read_excel(request.files["lead_data"])
        conversion_data = pd.read_excel(request.files["conversion_data"])

        # Step 3: Check if the number of rows in the first dataset is equal to the number of rows in the third dataset
        if lead_data.shape[0] != conversion_data.shape[0]:
            return "Error: Row count doesn't match between the first and third datasets."

        # Step 4: Convert all categorical columns in the first dataset into multiple columns using one-hot encoding
        categorical_columns = lead_data.select_dtypes(include=['object']).columns.tolist()
        lead_data_encoded = pd.get_dummies(lead_data, columns=categorical_columns, drop_first=True)

        # Step 5: Calculate p-value for correlation between each column of the first dataset and the 'Conversion' column
        final_columns = []
        for column in lead_data_encoded.columns:
            _, p_value = stats.pearsonr(lead_data_encoded[column], conversion_data['Conversion'])
            if p_value < 0.05:
                final_columns.append(column)

        # Creating the final dataset with significant columns
        final_data = lead_data_encoded[final_columns]

        # Step 6: Create a logistic regression model using the final dataset
        model = LogisticRegression()
        model.fit(final_data, conversion_data['Conversion'])

        # Step 7: Handle new lead data upload
        if "new_leads" not in request.files:
            return "Error: Please upload the new leads dataset."

        new_leads = pd.read_excel(request.files["new_leads"])

        # Check if columns in new leads dataset are same as those from the first dataset
        if set(new_leads.columns) != set(lead_data.columns):
            return "Error: Columns don't match in the new leads dataset."

        # Step 8: Predict the probability of conversion for each new lead
        new_leads_encoded = pd.get_dummies(new_leads, columns=categorical_columns, drop_first=True)

        # Ensure the columns match the trained model's columns
        missing_cols = set(final_data.columns) - set(new_leads_encoded.columns)
        for col in missing_cols:
            new_leads_encoded[col] = 0

        # Align the column order
        new_leads_encoded = new_leads_encoded[final_data.columns]

        # Predict probabilities
        predictions = model.predict_proba(new_leads_encoded)[:, 1]
        new_leads['Conversion_Probability'] = predictions

        # Step 9: Rank leads based on the predicted conversion probability
        new_leads['Rank'] = new_leads['Conversion_Probability'].rank(ascending=False)
        new_leads_sorted = new_leads.sort_values(by='Rank')

        # Output the ranked leads as HTML
        return new_leads_sorted.to_html()

    return render_template("index.html")

import os

if __name__ == "__main__":
    use_reloader = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=use_reloader)
